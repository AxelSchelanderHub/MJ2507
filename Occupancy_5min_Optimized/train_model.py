"""Development-only search; historical test evaluated after model selection."""
from pathlib import Path
import argparse
from dataclasses import asdict
import model_core as c
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
base_features = c.feature_matrix
def features(frame, family, minutes):
    clock = family.endswith('_clock')
    x = base_features(frame, family.replace('_clock', ''), minutes)
    if clock:
        hour = frame[c.TIME].dt.hour + frame[c.TIME].dt.minute / 60
        x = x.copy()
        x['hour_sin'] = np.sin(2*np.pi*hour/24)
        x['hour_cos'] = np.cos(2*np.pi*hour/24)
        x['weekend'] = (frame[c.TIME].dt.dayofweek >= 5).astype(float)
        x['night'] = ((hour >= 19) | (hour < 7)).astype(float)
    return x
c.feature_matrix = features

def select(raw, frame, p, config, start, end, **kwargs):
    ix = c.window(frame, start, end)
    y = frame.iloc[ix][c.TARGET].to_numpy()
    rows = []
    for threshold in np.arange(.05, .951, .01):
        pred = p[ix] >= threshold
        home = float(pred[y == 1].mean())
        away = float((~pred[y == 0]).mean())
        rows.append(dict(threshold=float(round(threshold, 2)), half_life=0, hysteresis=0.,
                         balanced_accuracy=(home+away)/2, home_recall=home, away_recall=away))
    best = max(rows, key=lambda r: (r['balanced_accuracy'], min(r['home_recall'],r['away_recall']), -abs(r['threshold']-.5)))
    return {k:best[k] for k in ('threshold','half_life','hysteresis')}, rows
c.select_policy = select

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE/'results')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    raw = c.load_data(HERE/'data/occupancy.csv')
    dev = raw[raw[c.TIME] < c.DEV_END].copy()
    recipes = [(8,.003,.001,1.,1.), (16,.003,.001,1.,1.),
               (32,.003,.01,1.,1.), (16,.001,.01,.5,1.),
               (16,.003,.001,1.,4.), (16,.01,.1,1.,1.)]
    configs = [c.Config(family=f, hidden=h,lr=lr,l2=l2,class_power=cw,transition_boost=b)
               for f in ('levels','levels_clock','dynamics','dynamics_clock')
               for h,lr,l2,cw,b in recipes]
    c.dump(out/'protocol.json',dict(objective='Mean development-fold balanced accuracy',
          configs=[asdict(x) for x in configs],folds=c.folds(),final_split=c.final_fold(),
          seeds=[42,7,21],thresholds='0.05 to 0.95 in 0.01 increments; calibration only',
          target='Independently recorded occupancy at each current five-minute timestamp',
          caveat='May 25–31 was previously inspected. Historical evaluation, not a fresh untouched test.'))
    rows=[]
    for cid, config in enumerate(configs):
        for split in c.folds():
            result,*_ = c.fit_fold(dev,config,split)
            rows.append(dict(config_id=cid,fold=split['fold'],**result))
        pd.DataFrame(rows).to_csv(out/'search_folds.csv',index=False)
        print(f'Completed candidate {cid+1}/{len(configs)}',flush=True)
    ranking=pd.DataFrame(rows).groupby('config_id').balanced_accuracy.mean().sort_values(ascending=False)
    ranking.to_csv(out/'search_ranking.csv')
    finals=[]
    for cid in ranking.index[:3]:
        for split in c.folds():
            result,*_ = c.fit_fold(dev,configs[cid],split,seeds=(42,7,21))
            finals.append(dict(config_id=int(cid),fold=split['fold'],**result))
        print(f'Ensemble finalist {cid} finished',flush=True)
    final_table=pd.DataFrame(finals)
    final_table.to_csv(out/'ensemble_folds.csv',index=False)
    winner=int(final_table.groupby('config_id').balanced_accuracy.mean().idxmax())
    config=configs[winner]
    c.dump(out/'locked_selection.json',dict(config_id=winner,config=asdict(config),
           validation_balanced_accuracy=float(final_table.groupby('config_id').balanced_accuracy.mean()[winner])))
    # Fit exclusively on development data; use calibration as a placeholder scoring window.
    split=c.final_fold()
    fit_split={**split,'score_start':split['cal_start'],'score_end':split['cal_end']}
    _,_,bundle,sweep,_=c.fit_fold(dev,config,fit_split,seeds=(42,7,21))
    c.save_bundle(out/'model_5min.npz',config,bundle)
    pd.DataFrame(sweep).to_csv(out/'validation_threshold_sweep.csv',index=False)
    c.dump(out/'final_parameters.json',dict(**asdict(config),seeds=[42,7,21],
           best_epochs=[m['best_epoch'] for m in bundle[0]],policy=bundle[1],features=bundle[2],
           model_sha256=c.sha(out/'model_5min.npz')))
    # Only now evaluate the historical test labels.
    pred=c.predict_saved(raw,out/'model_5min.npz')
    test=raw[[c.TIME,c.TARGET]].merge(pred,on=c.TIME)
    test=test[(test[c.TIME]>=c.TEST_START)&(test[c.TIME]<c.TEST_END)]
    test.to_csv(out/'test_predictions.csv',index=False)
    metrics=c.metrics(test[c.TARGET],test.predicted,test[c.TIME].to_numpy(),test.score_home)
    from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
    pr,re,f1,_=precision_recall_fscore_support(test[c.TARGET],test.predicted,labels=[0,1],zero_division=0)
    metrics.update(away_precision=pr[0],home_precision=pr[1],away_f1=f1[0],home_f1=f1[1],
                   macro_precision=float(pr.mean()),confusion_matrix=confusion_matrix(test[c.TARGET],test.predicted).tolist(),
                   test_samples=len(test),home_count=int(test[c.TARGET].sum()),threshold=bundle[1]['threshold'])
    c.dump(out/'test_metrics.json',metrics)
    print(metrics,flush=True)

if __name__=='__main__':
    main()
