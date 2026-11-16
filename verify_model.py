from pathlib import Path
import json
import argparse
import numpy as np
from train_model import c, select
import train_model
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--results-dir',type=Path,default=root/'results')
results=parser.parse_args().results_dir
protocol=json.loads((results/'protocol.json').read_text())
settings=protocol.get('split_settings',dict(stop_window_hours=48,gap_hours=24))
if (results/'resolved_config.json').exists():
    from configuration import make_splits
    resolved=json.loads((results/'resolved_config.json').read_text())
    all_folds,final_split=make_splits(resolved)
    train_model.THRESHOLDS=np.array(resolved['thresholds'])
else:
    all_folds,final_split=c.folds(**settings),c.final_fold(**settings)
raw=c.load_data(root/'data/occupancy.csv')
path=results/'model_5min.npz'
config,(models,policy,names)=c.load_bundle(path)
pred=c.predict_saved(raw,path)
saved=c.pd.read_csv(results/'test_predictions.csv')
test=pred[(pred[c.TIME]>=final_split['score_start'])&(pred[c.TIME]<final_split['score_end'])]
np.testing.assert_allclose(test.score_home,saved.score_home,atol=1e-12)
np.testing.assert_array_equal(test.predicted,saved.predicted)
# Changes to future sensors and all labels must not alter earlier predictions.
changed=raw.copy()
cut=len(raw)//2
changed.loc[cut:,c.SENSORS]=changed.loc[cut:,c.SENSORS]*9+100
changed[c.TARGET]=1-changed[c.TARGET]
altered=c.predict_saved(changed,path)
np.testing.assert_allclose(pred.score_home.iloc[:cut],altered.score_home.iloc[:cut])
np.testing.assert_allclose(c.predict_saved(raw.iloc[:cut],path).score_home,pred.score_home.iloc[:cut])
frame=c.aggregate(raw,5)
x=c.feature_matrix(frame,config.family,5,config)
train=c.window(frame,end=final_split['train_end'])
prep=c.fit_preprocessor(x.iloc[train],config.scale_floor)
for model in models:
    for key in prep:
        np.testing.assert_allclose(model['prep'][key],prep[key])
assert not any('Presence' in name or 'Humidity' in name for name in names)
for split in [*all_folds,final_split]:
    assert split['stop_start']-split['train_end']==c.pd.Timedelta(hours=settings['gap_hours'])
    assert split['cal_start']-split['stop_end']==c.pd.Timedelta(hours=settings['gap_hours'])
    assert split['score_start']-split['cal_end']==c.pd.Timedelta(hours=settings['gap_hours'])
    assert split['stop_end']-split['stop_start']==c.pd.Timedelta(hours=settings['stop_window_hours'])
m=json.loads((results/'test_metrics.json').read_text())
assert abs(m['balanced_accuracy']-(m['home_recall']+m['away_recall'])/2)<1e-12
split=final_split
prob=np.mean([c.probabilities(model,x,config.activation) for model in models],axis=0)
selected,_=select(raw,frame,prob,config,split['cal_start'],split['cal_end'])
assert selected==policy
print('PASS: saved prediction reproduction, future-sensor and label invariance, prefix invariance, training-only scaling, feature exclusions, temporal gaps, balanced accuracy.')
print('PASS: validation-only threshold reproduced.')
