from pathlib import Path
import json
import numpy as np
from train_model import c, select
root=Path(__file__).resolve().parent
raw=c.load_data(root/'data/occupancy.csv')
path=root/'results/model_5min.npz'
config,(models,policy,names)=c.load_bundle(path)
pred=c.predict_saved(raw,path)
saved=c.pd.read_csv(root/'results/test_predictions.csv')
test=pred[(pred[c.TIME]>=c.TEST_START)&(pred[c.TIME]<c.TEST_END)]
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
x=c.feature_matrix(frame,config.family,5)
train=c.window(frame,end=c.final_fold()['train_end'])
prep=c.fit_preprocessor(x.iloc[train])
for model in models:
    for key in prep:
        np.testing.assert_allclose(model['prep'][key],prep[key])
assert not any('Presence' in name or 'Humidity' in name for name in names)
for split in [*c.folds(),c.final_fold()]:
    assert split['stop_start']-split['train_end']>=c.pd.Timedelta(days=1)
    assert split['cal_start']-split['stop_end']>=c.pd.Timedelta(days=1)
    assert split['score_start']-split['cal_end']>=c.pd.Timedelta(days=1)
m=json.loads((root/'results/test_metrics.json').read_text())
assert abs(m['balanced_accuracy']-(m['home_recall']+m['away_recall'])/2)<1e-12
split=c.final_fold()
prob=np.mean([c.probabilities(model,x,config.activation) for model in models],axis=0)
selected,_=select(raw,frame,prob,config,split['cal_start'],split['cal_end'])
assert selected==policy
print('PASS: saved prediction reproduction, future-sensor and label invariance, prefix invariance, training-only scaling, feature exclusions, temporal gaps, balanced accuracy.')
print('PASS: validation-only threshold reproduced.')
