"""Causal, reproducible shallow-network occupancy study.

Search uses only data before May 24. The previously inspected May 25-31
period is a historical holdout, never a fresh independent test.
Python 3.10+; NumPy, pandas, scikit-learn, openpyxl, matplotlib.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
from pathlib import Path
import argparse, hashlib, importlib.util, json, time, warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
TIME = 'Timestep'
TARGET = 'Presence (1 = presence)'
SENSORS = ['CO2 (ppm)', 'Light (lux)', 'Indoor temperature (°C)', 'Outdoor temperature (°C)']
HUMIDITY = 'Relative Humidity (%)'
DEV_END = pd.Timestamp('2020-05-24')
TEST_START = pd.Timestamp('2020-05-25')
TEST_END = pd.Timestamp('2020-06-01')
SEEDS = (42, 7, 21)
THRESHOLDS = np.round(np.arange(.10, .901, .025), 3)
HALF_LIVES = (0, 5, 15)
HYSTERESIS = (0., .05)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x), allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path, labels=True):
    path = Path(path)
    if path.suffix.lower() == '.xlsx':
        df = pd.read_excel(path, sheet_name='FINAL CLEAN DATA TO DOWNLOAD', engine='openpyxl')
    else:
        df = pd.read_csv(path)
    required = [TIME, *SENSORS] + ([TARGET] if labels else [])
    if not set(required).issubset(df.columns):
        raise ValueError(f'Missing columns: {set(required)-set(df.columns)}')
    df[TIME] = pd.to_datetime(df[TIME], format='mixed', errors='raise')
    if df[TIME].isna().any() or df[TIME].duplicated().any() or not df[TIME].is_monotonic_increasing:
        raise ValueError('Timestamps must be valid, unique and increasing; sort/resolve duplicates upstream.')
    for c in [*SENSORS, HUMIDITY]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if labels:
        if not df[TARGET].isin([0, 1]).all():
            raise ValueError('Independently recorded binary labels are required.')
        df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)


def aggregate(raw, minutes):
    """Non-overlapping right-closed means, available at actual last reading.

    Subsecond Excel clock drift is used only to identify nominal five-minute
    slots. Availability remains the original (unrounded) timestamp.
    Incomplete intervals are discarded, never filled from future readings.
    Occupancy target is the LAST true state, not a majority/mean label.
    """
    if minutes not in (5, 15, 30):
        raise ValueError('Supported aggregation periods: 5, 15, 30 minutes')
    raw = raw.copy()
    nominal = raw[TIME].dt.round('5min')
    if ((raw[TIME]-nominal).abs() > pd.Timedelta(seconds=1)).any():
        raise ValueError('Input must follow a five-minute grid within one second.')
    if nominal.duplicated().any():
        raise ValueError('Multiple readings map to one nominal five-minute slot.')
    raw['_nominal'] = nominal
    raw['_bin'] = nominal.dt.ceil(f'{minutes}min')
    g = raw.groupby('_bin', sort=True)
    cols = [c for c in [*SENSORS, HUMIDITY] if c in raw]
    means = g[cols].mean()
    means[TIME] = g[TIME].max()
    means['_source_min'] = g[TIME].min()
    means['_source_max'] = g[TIME].max()
    means['_nominal_end'] = g['_nominal'].max()
    means['_count'] = g.size()
    if TARGET in raw:
        means[TARGET] = g[TARGET].last()
        means['_home_fraction'] = g[TARGET].mean()  # diagnostic ONLY, never a feature
    stds = g[SENSORS].std(ddof=0).add_suffix('_within_std')
    means = means.join(stds)
    full = (means['_count'] == minutes//5) & (means['_nominal_end'] == means.index)
    means = means.loc[full].reset_index(drop=True)
    if not (means['_source_max'] <= means[TIME]).all():
        raise AssertionError('Future readings entered an aggregate.')
    return means


def feature_matrix(frame, family, minutes, config=None):
    """Sensors and optional timestamp features only. No target-derived input."""
    if family not in ('levels', 'dynamics', 'dynamics_clock'):
        raise ValueError(family)
    config = config or Config()
    x = frame[SENSORS].copy().astype(float)
    if family == 'levels':
        return x
    x['Light (lux)'] = np.log1p(np.maximum(x['Light (lux)'], 0))
    parts = [x.add_suffix('_level')]
    # Segment history at missing intervals; no history crosses a data outage.
    times = pd.DatetimeIndex(frame[TIME])
    breaks = np.r_[True, np.diff(times.as_unit('ns').asi8) > (minutes*60+1)*1e9]
    segment = np.cumsum(breaks)
    group = x.groupby(segment)
    for lag in sorted(set(max(1, int(lag)//minutes) for lag in config.delta_minutes)):
        parts.append((x - group.shift(lag)).add_suffix(f'_delta_{lag*minutes}m'))
    k = max(2, config.history_minutes//minutes)
    past_mean = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    past_std = past_mean.copy()
    for _, index in pd.Series(np.arange(len(x))).groupby(segment):
        indices = index.to_numpy()
        past = x.iloc[indices].shift(1).rolling(k, min_periods=1)
        past_mean.iloc[indices] = past.mean().to_numpy()
        past_std.iloc[indices] = past.std(ddof=0).to_numpy()
    parts += [past_mean.add_suffix('_past_mean'), past_std.add_suffix('_past_std'),
              (x-past_mean).add_suffix('_deviation'), frame[[c+'_within_std' for c in SENSORS]].copy()]
    parts.append(pd.DataFrame({'indoor_minus_outdoor': frame[SENSORS[2]]-frame[SENSORS[3]]}))
    if family == 'dynamics_clock':
        hour = frame[TIME].dt.hour + frame[TIME].dt.minute/60
        parts.append(pd.DataFrame({'hour_sin': np.sin(2*np.pi*hour/24),
                                  'hour_cos': np.cos(2*np.pi*hour/24),
                                  'weekend': (frame[TIME].dt.dayofweek >= 5).astype(float)}))
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


def fit_preprocessor(x, scale_floor=1e-8):
    x = np.asarray(x, float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.)
    filled = np.where(np.isfinite(x), x, med)
    mean, std = filled.mean(axis=0), filled.std(axis=0)
    std[std < scale_floor] = 1.
    return {'medians': med, 'mean': mean, 'std': std}


def transform(x, prep):
    x = np.asarray(x, float)
    return (np.where(np.isfinite(x), x, prep['medians'])-prep['mean'])/prep['std']


@dataclass(frozen=True)
class Config:
    minutes: int = 5
    family: str = 'levels'
    hidden: int = 16
    activation: str = 'tanh'
    optimizer: str = 'adam'
    lr: float = .003
    l2: float = .001
    class_power: float = 1.
    transition_boost: float = 1.
    epochs: int = 900
    check_every: int = 10
    patience_checks: int = 15
    min_delta: float = 1e-5
    adam_beta1: float = .9
    adam_beta2: float = .999
    adam_gradient_weight1: float = .1
    adam_gradient_weight2: float = .001
    adam_epsilon: float = 1e-8
    init_scale: float = 1.
    transition_minutes: float = 15.
    transition_max_gap_minutes: float = 31.
    history_minutes: int = 60
    delta_minutes: tuple = (5,30,60)
    night_start_hour: float = 19.
    night_end_hour: float = 7.
    scale_floor: float = 1e-8


def initialize(n_features, config, seed):
    rng = np.random.default_rng(seed)
    return dict(W1=rng.normal(0, config.init_scale*np.sqrt(1/n_features), (n_features, config.hidden)),
                b1=np.zeros(config.hidden), W2=rng.normal(0, config.init_scale*np.sqrt(1/config.hidden), config.hidden), b2=np.zeros(1))


def forward(x, params, activation):
    z = x @ params['W1'] + params['b1']
    h = np.tanh(z) if activation == 'tanh' else np.maximum(z, 0)
    logits = h @ params['W2'] + params['b2'][0]
    p = np.exp(-np.logaddexp(0, -logits))
    return p, (h, logits, z)


def loss_grad(x, y, weights, params, config):
    p, (h, z, z1) = forward(x, params, config.activation)
    loss = np.mean(weights*(np.logaddexp(0, z)-y*z)) + .5*config.l2*(np.sum(params['W1']**2)+np.sum(params['W2']**2))
    dz = weights*(p-y)/len(y)
    dh = dz[:, None]*params['W2']
    dh *= (1-h*h) if config.activation == 'tanh' else (z1 > 0)
    grads = dict(W1=x.T@dh + config.l2*params['W1'], b1=dh.sum(axis=0),
                 W2=h.T@dz + config.l2*params['W2'], b2=np.array([dz.sum()]))
    return float(loss), grads


def class_weights(y, power):
    counts = np.bincount(np.asarray(y, int), minlength=2)
    if (counts == 0).any():
        raise ValueError('Training must contain both occupancy states.')
    cw = (len(y)/(2*counts))**power
    return cw/np.mean(cw[np.asarray(y, int)])


def training_weights(y, times, cw, boost, transition_minutes=15., max_gap_minutes=31.):
    """Post-change training emphasis uses training labels, never feature inputs."""
    w = cw[np.asarray(y, int)].copy()
    if boost > 1:
        t = pd.DatetimeIndex(times).as_unit('ns').asi8/60e9
        last = -np.inf
        for i in range(1, len(y)):
            if y[i] != y[i-1] and t[i]-t[i-1] <= max_gap_minutes:
                last = t[i]
            if t[i]-last <= transition_minutes:
                w[i] *= boost
    return w/w.mean()


def train_nn(x, y, times, xv, yv, config, seed):
    prep = fit_preprocessor(x, config.scale_floor)
    x, xv = transform(x, prep), transform(xv, prep)
    y, yv = np.asarray(y, float), np.asarray(yv, float)
    cw = class_weights(y, config.class_power)
    w = training_weights(y, times, cw, config.transition_boost, config.transition_minutes, config.transition_max_gap_minutes)
    # Stopping loss balances classes using TRAINING counts; no transition labels.
    wv = cw[yv.astype(int)]
    params = initialize(x.shape[1], config, seed)
    first = {k: np.zeros_like(v) for k, v in params.items()}
    second = {k: np.zeros_like(v) for k, v in params.items()}
    best = {k: v.copy() for k, v in params.items()}
    best_loss, stale, best_epoch = float('inf'), 0, 0
    history = []
    for epoch in range(1, config.epochs+1):
        train_loss, gradients = loss_grad(x, y, w, params, config)
        for key, gradient in gradients.items():
            if config.optimizer == 'adam':
                first[key] = config.adam_beta1*first[key]+config.adam_gradient_weight1*gradient
                second[key] = config.adam_beta2*second[key]+config.adam_gradient_weight2*gradient*gradient
                params[key] -= config.lr*(first[key]/(1-config.adam_beta1**epoch))/(np.sqrt(second[key]/(1-config.adam_beta2**epoch))+config.adam_epsilon)
            else:
                params[key] -= config.lr*gradient
        if epoch == 1 or epoch % config.check_every == 0:
            _, (_, logits, _) = forward(xv, params, config.activation)
            val_loss = float(np.mean(wv*(np.logaddexp(0, logits)-yv*logits)))
            history.append(dict(epoch=epoch, train_loss=train_loss, stop_loss=val_loss))
            if not np.isfinite(val_loss):
                raise ValueError('Non-finite training loss')
            if val_loss < best_loss-config.min_delta:
                best_loss, stale, best_epoch = val_loss, 0, epoch
                best = {k: v.copy() for k, v in params.items()}
            else:
                stale += 1
            if stale >= config.patience_checks:
                break
    return dict(parameters=best, prep=prep, history=history, best_epoch=best_epoch, class_weights=cw)


def probabilities(model, x, activation):
    return forward(transform(x, model['prep']), model['parameters'], activation)[0]


def temporal_split(start, end, number, stop_window_hours=48, gap_hours=24, calibration_hours=48):
    if not np.isfinite(stop_window_hours) or stop_window_hours <= 0:
        raise ValueError('stop_window_hours must be positive and finite')
    if not np.isfinite(gap_hours) or gap_hours < 0:
        raise ValueError('gap_hours must be nonnegative and finite')
    s = pd.Timestamp(start)
    gap = pd.Timedelta(hours=gap_hours)
    cal_end = s-gap
    cal_start = cal_end-pd.Timedelta(hours=calibration_hours)
    stop_end = cal_start-gap
    stop_start = stop_end-pd.Timedelta(hours=stop_window_hours)
    return dict(fold=number, train_end=stop_start-gap,
                stop_start=stop_start, stop_end=stop_end,
                cal_start=cal_start, cal_end=cal_end,
                score_start=s, score_end=pd.Timestamp(end))


def folds(stop_window_hours=48, gap_hours=24):
    # All windows use exclusive right ends. All three gaps are 24 hours.
    result = []
    for number, start in enumerate(('2020-05-14', '2020-05-17', '2020-05-21'), 1):
        s = pd.Timestamp(start)
        result.append(temporal_split(s, s+pd.Timedelta(days=3), number,
                                     stop_window_hours, gap_hours))
    return result


def final_fold(stop_window_hours=48, gap_hours=24):
    return temporal_split(TEST_START, TEST_END, 4, stop_window_hours, gap_hours)


def window(df, start=None, end=None):
    keep = np.ones(len(df), dtype=bool)
    if start is not None:
        keep &= df[TIME].to_numpy() >= np.datetime64(start)
    if end is not None:
        keep &= df[TIME].to_numpy() < np.datetime64(end)
    return np.flatnonzero(keep)


def filter_scores(p, times, half_life):
    p = np.asarray(p, float)
    out = p.copy()
    if not half_life:
        return out
    dt = np.diff(pd.DatetimeIndex(times).as_unit('ns').asi8)/60e9
    for i in range(1, len(p)):
        if dt[i-1] <= 31:
            a = 1-np.exp(-np.log(2)*dt[i-1]/half_life)
            out[i] = a*p[i]+(1-a)*out[i-1]
    return out


def classify(p, times, threshold, half_life=0, hysteresis=0):
    smooth = filter_scores(p, times, half_life)
    pred = (smooth >= threshold).astype(int)
    if hysteresis:
        dt = np.diff(pd.DatetimeIndex(times).as_unit('ns').asi8)/60e9
        for i in range(1, len(pred)):
            if dt[i-1] > 31:
                continue
            cutoff = threshold-hysteresis if pred[i-1] else threshold+hysteresis
            pred[i] = int(smooth[i] >= cutoff)
    return pred, smooth


def align_to_raw(raw, frame_times, values, minutes):
    """Most recent AVAILABLE prediction, expiring after one update interval."""
    rt = pd.DatetimeIndex(raw[TIME]).as_unit('ns').asi8
    ft = pd.DatetimeIndex(frame_times).as_unit('ns').asi8
    j = np.searchsorted(ft, rt, side='right')-1
    good = j >= 0
    age = np.full(len(rt), np.inf)
    age[good] = (rt[good]-ft[j[good]])/60e9
    good &= age < minutes+.02
    # All scored windows have preceding sensor context, so coverage is full.
    out = np.full(len(rt), np.nan)
    out[good] = np.asarray(values)[j[good]]
    return out, good


def event_details(y, pred, times, tolerance=30):
    """One-to-one causal matches, before the next opposite true state.

    A departure alarm after the person has already returned earns no credit.
    Early alarms earn no credit. Changes across gaps are not timed events.
    """
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    t = pd.DatetimeIndex(times).as_unit('ns').asi8/60e9
    continuous = np.r_[False, np.diff(t) <= 5.02]
    truth = np.flatnonzero(np.r_[False, y[1:] != y[:-1]] & continuous)
    guesses = np.flatnonzero(np.r_[False, pred[1:] != pred[:-1]] & continuous)
    used, rows = set(), []
    for k, i in enumerate(truth):
        next_change = t[truth[k+1]] if k+1 < len(truth) else np.inf
        # Never bridge a later observation gap.
        future_gap = np.flatnonzero(~continuous[i+1:])
        if len(future_gap):
            next_change = min(next_change, t[i+1+future_gap[0]])
        eligible = [j for j in guesses if j not in used and pred[j] == y[i]
                    and t[j] >= t[i] and t[j]-t[i] <= tolerance+.01 and t[j] < next_change]
        match = eligible[0] if eligible else None
        if match is not None:
            used.add(match)
        rows.append(dict(time=str(pd.Timestamp(times[i])), state=int(y[i]), detected=int(match is not None),
                         delay_minutes=float(t[match]-t[i]) if match is not None else None))
    false = [dict(time=str(pd.Timestamp(times[j])), state=int(pred[j])) for j in guesses if j not in used]
    return rows, false


def metric_counts(y, pred, times):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    ev, false = event_details(y, pred, times)
    arrivals = sum(r['state'] == 1 for r in ev)
    atp = sum(r['state'] == 1 and r['detected'] for r in ev)
    departures = len(ev)-arrivals
    dtp = sum(r['state'] == 0 and r['detected'] for r in ev)
    deltas = np.diff(pd.DatetimeIndex(times).as_unit('ns').asi8)/60e9
    days = float((np.minimum(deltas, 5).sum()+5)/1440)
    delays = [r['delay_minutes'] for r in ev if r['detected']]
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, arrivals=arrivals, departures=departures,
                arrivals_detected=atp, departures_detected=dtp, false_events=len(false), observed_days=days,
                median_delay_minutes=float(np.median(delays)) if delays else None)


def rates(c):
    div = lambda a, b: float(a/b) if b else 0.
    tp, tn, fp, fn = (c[k] for k in ('tp', 'tn', 'fp', 'fn'))
    home, away = div(tp,tp+fn), div(tn,tn+fp)
    etp = c['arrivals_detected']+c['departures_detected']
    en = c['arrivals']+c['departures']
    efp = c['false_events']
    ef2 = div(5*etp,5*etp+4*(en-etp)+efp)
    return dict(accuracy=div(tp+tn,tp+tn+fp+fn), home_recall=home, away_recall=away,
                home_precision=div(tp,tp+fp), away_precision=div(tn,tn+fn), balanced_accuracy=(home+away)/2,
                macro_f1=(div(2*tp,2*tp+fp+fn)+div(2*tn,2*tn+fp+fn))/2,
                event_recall=div(etp,en), event_precision=div(etp,etp+efp), event_f2=ef2,
                arrival_recall=div(c['arrivals_detected'],c['arrivals']), departure_recall=div(c['departures_detected'],c['departures']),
                false_events_per_day=div(efp,c['observed_days']), utility=.5*((home+away)/2)+.5*ef2)


def metrics(y, pred, times, scores=None):
    c = metric_counts(y,pred,times)
    out = dict(**c, **rates(c))
    if scores is not None and len(np.unique(y)) == 2:
        out.update(roc_auc=float(roc_auc_score(y,scores)),
                   average_precision_away=float(average_precision_score(1-np.asarray(y),1-np.asarray(scores))))
    return out


def predictions_for_window(raw, frame, p, config, policy, start, end):
    # Warm up postprocessing on earlier unlabeled sensor predictions.
    pred, smooth = classify(p, frame[TIME], **policy)
    r = raw.iloc[window(raw,start,end)].copy()
    aligned, good = align_to_raw(r,frame[TIME],pred,config.minutes)
    prob, good2 = align_to_raw(r,frame[TIME],smooth,config.minutes)
    good &= good2
    if not good.all():
        raise ValueError(f'Incomplete prediction coverage in scored window: {int((~good).sum())}')
    r['predicted'] = aligned.astype(int)
    r['score_home'] = prob
    return r[[TIME,TARGET,'predicted','score_home']]


def select_policy(raw, frame, p, config, start, end, macro_only=False):
    best, key_best, rows = None, None, []
    half_lives = (0,) if macro_only else HALF_LIVES
    hyst = (0,) if macro_only else HYSTERESIS
    # Use the same complete causal warmup as inference, without its labels.
    fi = window(frame,end=end)
    f, p = frame.iloc[fi], np.asarray(p)[fi]
    r = raw.iloc[window(raw,start,end)]
    for hl in half_lives:
        for h in hyst:
            for threshold in THRESHOLDS:
                policy = dict(threshold=float(threshold),half_life=int(hl),hysteresis=float(h))
                pred, _ = classify(p,f[TIME],**policy)
                aligned, good = align_to_raw(r,f[TIME],pred,config.minutes)
                if not good.all():
                    raise ValueError('Calibration prediction coverage failed')
                m = metrics(r[TARGET].to_numpy(),aligned.astype(int),r[TIME].to_numpy())
                objective = m['macro_f1'] if macro_only else m['utility']
                key = (objective, min(m['home_recall'],m['away_recall']), -m['false_events_per_day'], -abs(threshold-.5), -hl, -h)
                rows.append(dict(**policy,**m))
                if key_best is None or key > key_best:
                    best, key_best = policy, key
    return best, rows


def fit_fold(raw, config, split, seeds=(42,), save_history=False):
    frame = aggregate(raw,config.minutes)
    features = feature_matrix(frame,config.family,config.minutes,config)
    train = window(frame,end=split['train_end'])
    stop = window(frame,split['stop_start'],split['stop_end'])
    if len(train) == 0 or len(stop) == 0:
        raise ValueError('Empty training or stopping window')
    models, all_p = [], []
    for seed in seeds:
        model = train_nn(features.iloc[train],frame.iloc[train][TARGET].to_numpy(),frame.iloc[train][TIME],
                         features.iloc[stop],frame.iloc[stop][TARGET].to_numpy(),config,seed)
        models.append(model)
        all_p.append(probabilities(model,features,config.activation))
    p = np.mean(all_p,axis=0)
    policy, sweep = select_policy(raw,frame,p,config,split['cal_start'],split['cal_end'])
    predictions = predictions_for_window(raw,frame,p,config,policy,split['score_start'],split['score_end'])
    result = metrics(predictions[TARGET],predictions.predicted,predictions[TIME].to_numpy(),predictions.score_home)
    result.update(**policy,best_epoch=float(np.mean([m['best_epoch'] for m in models])), n_train=len(train), n_stop=len(stop), n_features=features.shape[1])
    return result, predictions, (models,policy,list(features.columns)), sweep, (frame,features,np.array(all_p))


def candidate_configs():
    # Predeclared 16 recipes, crossed with 3 time frames and 3 feature families.
    # A designed search, not every possible combination of these parameters.
    recipes = [
        (8,'tanh','adam',.003,.001,1.,1.),
        (16,'tanh','adam',.003,.001,1.,1.),
        (32,'tanh','adam',.003,.001,1.,1.),
        (64,'tanh','adam',.003,.01,1.,1.),
        (16,'tanh','adam',.001,.0001,.5,1.),
        (16,'tanh','adam',.01,.01,1.,1.),
        (16,'tanh','adam',.003,.1,1.,1.),
        (32,'tanh','adam',.001,.01,1.,2.),
        (16,'tanh','adam',.003,.001,1.,4.),
        (16,'tanh','adam',.003,.001,.5,2.),
        (16,'tanh','adam',.003,.001,0.,1.),
        (32,'relu','adam',.003,.001,1.,1.),
        (16,'relu','adam',.001,.01,.5,2.),
        (32,'relu','adam',.01,.01,1.,4.),
        (12,'tanh','gd',.1,.001,1.,1.),
        (24,'tanh','gd',.03,.01,1.,2.),
    ]
    return [Config(minutes,family,*recipe) for minutes in (5,15,30)
            for family in ('levels','dynamics','dynamics_clock') for recipe in recipes]


def rank_results(rows):
    df = pd.DataFrame(rows)
    keys = ['utility','balanced_accuracy','home_recall','away_recall','event_f2','event_recall','event_precision','false_events_per_day','accuracy','macro_f1']
    mean = df.groupby('config_id')[keys].mean()
    mean['utility_sd'] = df.groupby('config_id').utility.std().fillna(0)
    mean['selection_score'] = mean.utility-.10*mean.utility_sd
    return mean.sort_values(['selection_score','balanced_accuracy'],ascending=False).reset_index()


def search_job(raw,config,split):
    return fit_fold(raw,config,split)[0]


def run_search(data, out, smoke=False):
    if (data[TIME] >= DEV_END).any():
        raise ValueError('run_search accepts DEVELOPMENT data only (before May 24).')
    out.mkdir(parents=True,exist_ok=True)
    configs = candidate_configs()
    if smoke:
        configs = configs[:1]
    dump(out/'search_protocol.json',dict(configurations=[asdict(c) for c in configs], folds=folds(),
        target='Current independently recorded occupancy, 1=home', exclude_humidity=True,
        score='0.5 balanced accuracy + 0.5 strict transition F2; rank mean minus 0.10 fold SD',
        thresholds=THRESHOLDS.tolist(), half_lives=HALF_LIVES,hysteresis=HYSTERESIS,
        final_holdout_start=TEST_START,development_end=DEV_END,final_fit=final_fold(),
        seeds=SEEDS,finalists_per_frame=2, historical_holdout_already_inspected=True,
        script_sha256=sha(__file__)))
    rows = []
    started = time.perf_counter()
    cache_path = out/'search_folds.csv'
    done = set()
    if cache_path.exists() and not smoke:
        rows = pd.read_csv(cache_path).to_dict('records')
        done = {(int(r['config_id']),int(r['fold'])) for r in rows}
    with ProcessPoolExecutor(max_workers=3) as executor:
        for cid, config in enumerate(configs):
            jobs = [(split,executor.submit(search_job,data,config,split)) for split in folds() if (cid,split['fold']) not in done]
            for split,future in jobs:
                result = future.result()
                rows.append(dict(config_id=cid,fold=split['fold'],seed=42,**asdict(config),**result))
                pd.DataFrame(rows).to_csv(cache_path,index=False)
            last = [r for r in rows if r['config_id']==cid]
            print(f'SEARCH {cid+1}/{len(configs)} {config.minutes}min {config.family} utility={np.mean([r["utility"] for r in last]):.3f} elapsed={time.perf_counter()-started:.0f}s',flush=True)
    ranking = rank_results(rows)
    ranking = ranking.merge(pd.DataFrame([dict(config_id=i,**asdict(c)) for i,c in enumerate(configs)]),on='config_id')
    ranking.to_csv(out/'search_summary.csv',index=False)
    if smoke:
        return
    finalists = ranking.groupby('minutes',sort=True).head(2).config_id.tolist()
    ensemble_rows, seed_rows, oof_rows = [],[],[]
    for cid in finalists:
        config = configs[cid]
        for split in folds():
            result, pred, bundle, _, detail = fit_fold(data,config,split,seeds=SEEDS)
            ensemble_rows.append(dict(config_id=cid,fold=split['fold'],**asdict(config),**result))
            oof_rows.append(pred.assign(config_id=cid,fold=split['fold']))
            frame,_,all_p = detail
            for seed,p in zip(SEEDS,all_p):
                policy,_ = select_policy(data,frame,p,config,split['cal_start'],split['cal_end'])
                s = predictions_for_window(data,frame,p,config,policy,split['score_start'],split['score_end'])
                seed_rows.append(dict(config_id=cid,fold=split['fold'],seed=seed,**metrics(s[TARGET],s.predicted,s[TIME].to_numpy(),s.score_home)))
        pd.DataFrame(ensemble_rows).to_csv(out/'ensemble_folds.csv',index=False)
        pd.DataFrame(seed_rows).to_csv(out/'seed_sensitivity.csv',index=False)
        print(f'ENSEMBLE config {cid} completed',flush=True)
    erank = rank_results(ensemble_rows).merge(pd.DataFrame([dict(config_id=i,**asdict(c)) for i,c in enumerate(configs)]),on='config_id')
    erank.to_csv(out/'ensemble_summary.csv',index=False)
    pd.concat(oof_rows).to_csv(out/'finalist_validation_predictions.csv',index=False)
    frame_winners = erank.groupby('minutes',sort=True).head(1).config_id.astype(int).tolist()
    winner = int(erank.iloc[0].config_id)
    selection = dict(winner=winner, frame_winners=frame_winners, candidates={str(cid):asdict(configs[cid]) for cid in frame_winners},
                     seeds=SEEDS,chosen_using='earlier validation folds only',
                     selection_score=float(erank.iloc[0].selection_score),
                     source_sha256=sha(HERE/'data'/'occupancy.csv'), protocol_sha256=sha(out/'search_protocol.json'),
                     final_holdout_used_for_selection=False, historical_holdout_already_inspected=True)
    dump(out/'locked_selection.json',selection)
    print('LOCKED SELECTION',json.dumps(selection),flush=True)


def save_bundle(path, config, bundle):
    models,policy,features = bundle
    values = dict(config_json=np.array(json.dumps(asdict(config))),policy_json=np.array(json.dumps(policy)),feature_names=np.array(features),n_models=np.array(len(models)))
    for i,model in enumerate(models):
        for k,v in model['parameters'].items():
            values[f'{i}_{k}'] = v
        for k,v in model['prep'].items():
            values[f'{i}_{k}'] = v
    np.savez_compressed(path,**values)


def load_bundle(path):
    with np.load(path,allow_pickle=False) as z:
        config = Config(**json.loads(str(z['config_json'])))
        policy = json.loads(str(z['policy_json']))
        models = [dict(parameters={k:z[f'{i}_{k}'].copy() for k in ('W1','b1','W2','b2')},
                       prep={k:z[f'{i}_{k}'].copy() for k in ('medians','mean','std')}) for i in range(int(z['n_models']))]
        features = z['feature_names'].tolist()
    return config,(models,policy,features)


def predict_saved(raw, path):
    config,(models,policy,names) = load_bundle(path)
    frame = aggregate(raw,config.minutes)
    features = feature_matrix(frame,config.family,config.minutes,config)
    if list(features.columns) != names:
        raise ValueError('Saved model feature schema mismatch')
    p = np.mean([probabilities(m,features,config.activation) for m in models],axis=0)
    pred,smooth = classify(p,frame[TIME],**policy)
    result = frame[[TIME]].copy()
    result['score_home'] = smooth
    result['predicted'] = pred
    result['threshold'] = policy['threshold']
    result['history_minutes'] = np.minimum((result[TIME]-result[TIME].iloc[0]).dt.total_seconds()/60,120)
    return result


def original_fold(raw, split, version, humidity=True, preset=False):
    spec = importlib.util.spec_from_file_location(f'original{version}',HERE/'reference'/f'network{version}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.FEATURE_COLS = [*SENSORS]+([HUMIDITY] if humidity else [])
    tr = raw.iloc[window(raw,end=split['train_end'])]
    st = raw.iloc[window(raw,split['stop_start'],split['stop_end'])]
    prep = m.fit_preprocessor(tr)
    kwargs = dict(max_epochs=1800,patience_checks=35)
    if version == 6:
        kwargs['verbose'] = False
    if preset:
        kwargs.update(n_h=12,learning_rate=.15,l2_lambda=10.,max_epochs=900,patience_checks=15)
    params,history,_ = m.nn_model(m.transform_x(tr,prep),m.labels(tr),m.transform_x(st,prep),m.labels(st),**kwargs)
    p = m.predict_proba(params,m.transform_x(raw,prep))
    if preset:
        policy,_ = select_policy(raw,raw,p,Config(),split['cal_start'],split['cal_end'])
        pred,smooth = classify(p,raw[TIME],**policy)
    else:
        cal = window(raw,split['cal_start'],split['cal_end'])
        threshold = m.choose_threshold(raw.iloc[cal][TARGET].to_numpy(),p[cal])
        policy = dict(threshold=threshold,half_life=0,hysteresis=0.)
        pred,smooth = classify(p,raw[TIME],**policy)
    score = window(raw,split['score_start'],split['score_end'])
    r = raw.iloc[score][[TIME,TARGET]].copy()
    r['predicted'],r['score_home'] = pred[score],smooth[score]
    return metrics(r[TARGET],r.predicted,r[TIME].to_numpy(),r.score_home),r,policy


def run_baselines(data,out,include_holdout=False):
    rows, predictions = [],[]
    split_list = [final_fold()] if include_holdout else folds()
    for split in split_list:
        for name,version,humidity,preset in [('Network 4 (humidity)',4,True,False),('Network 6 (humidity)',6,True,False),
                                             ('Network 6 (no humidity)',6,False,False),('Network 6 recall preset (no humidity)',6,False,True)]:
            result,pred,policy = original_fold(data,split,version,humidity,preset)
            rows.append(dict(model=name,fold=split['fold'],**result,**policy))
            predictions.append(pred.assign(model=name,fold=split['fold']))
        config = Config()
        frame = aggregate(data,5)
        x = feature_matrix(frame,'levels',5)
        train = window(frame,end=split['train_end'])
        prep = fit_preprocessor(x.iloc[train])
        logit = LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=42)
        logit.fit(transform(x.iloc[train],prep),frame.iloc[train][TARGET])
        p = logit.predict_proba(transform(x,prep))[:,1]
        policy,_ = select_policy(data,frame,p,config,split['cal_start'],split['cal_end'])
        pred = predictions_for_window(data,frame,p,config,policy,split['score_start'],split['score_end'])
        result = metrics(pred[TARGET],pred.predicted,pred[TIME].to_numpy(),pred.score_home)
        rows.append(dict(model='Logistic regression (no humidity)',fold=split['fold'],**result,**policy))
        predictions.append(pred.assign(model='Logistic regression (no humidity)',fold=split['fold']))
        pred = data.iloc[window(data,split['score_start'],split['score_end'])][[TIME,TARGET]].copy()
        pred['predicted'],pred['score_home'] = 1,1.
        result = metrics(pred[TARGET],pred.predicted,pred[TIME].to_numpy(),pred.score_home)
        rows.append(dict(model='Always home',fold=split['fold'],**result))
        predictions.append(pred.assign(model='Always home',fold=split['fold']))
        print('BASELINES fold',split['fold'],'completed',flush=True)
    suffix = 'historical' if include_holdout else 'validation'
    pd.DataFrame(rows).to_csv(out/f'baselines_{suffix}.csv',index=False)
    pd.concat(predictions).to_csv(out/f'baseline_predictions_{suffix}.csv',index=False)
    return rows


def prepare_final(data,out):
    """Fit and calibrate all frozen finalists WITHOUT reading holdout labels."""
    selection = json.loads((out/'locked_selection.json').read_text())
    if (data[TIME] >= DEV_END).any():
        raise ValueError('Final preparation must receive development data only.')
    split = final_fold()
    trained = {}
    for cid in selection['frame_winners']:
        config = Config(**selection['candidates'][str(cid)])
        frame = aggregate(data,config.minutes)
        features = feature_matrix(frame,config.family,config.minutes,config)
        train, stop = window(frame,end=split['train_end']),window(frame,split['stop_start'],split['stop_end'])
        models = [train_nn(features.iloc[train],frame.iloc[train][TARGET].to_numpy(),frame.iloc[train][TIME],
                           features.iloc[stop],frame.iloc[stop][TARGET].to_numpy(),config,seed) for seed in SEEDS]
        p = np.mean([probabilities(m,features,config.activation) for m in models],axis=0)
        policy,sweep = select_policy(data,frame,p,config,split['cal_start'],split['cal_end'])
        path = out/f'model_{config.minutes}min.npz'
        save_bundle(path,config,(models,policy,list(features.columns)))
        pd.DataFrame(sweep).to_csv(out/f'calibration_sweep_{config.minutes}min.csv',index=False)
        dump(out/f'training_history_{config.minutes}min.json',[dict(seed=s,best_epoch=m['best_epoch'],history=m['history']) for s,m in zip(SEEDS,models)])
        trained[str(cid)] = dict(config=asdict(config),policy=policy,model_path=path.name,sha256=sha(path),best_epochs=[m['best_epoch'] for m in models])
    dump(out/'frozen_models.json',dict(selection_sha256=sha(out/'locked_selection.json'),models=trained,holdout_accessed=False))
    print('ALL FINAL MODELS FROZEN; historical holdout has not entered fitting/calibration.',flush=True)


def evaluate_final(data,out):
    if (out/'historical_metrics.csv').exists():
        raise RuntimeError('Historical evaluation already exists. Do not tune and reopen the same holdout.')
    frozen = json.loads((out/'frozen_models.json').read_text())
    selection = json.loads((out/'locked_selection.json').read_text())
    if frozen['selection_sha256'] != sha(out/'locked_selection.json'):
        raise ValueError('Selection lock changed after fitting.')
    rows,predictions = [],[]
    for cid,item in frozen['models'].items():
        path = out/item['model_path']
        if sha(path) != item['sha256']:
            raise ValueError('Saved model changed after freezing.')
        config,bundle = load_bundle(path)
        models,policy,names = bundle
        frame = aggregate(data,config.minutes)
        x = feature_matrix(frame,config.family,config.minutes,config)
        p = np.mean([probabilities(m,x,config.activation) for m in models],axis=0)
        r = predictions_for_window(data,frame,p,config,policy,TEST_START,TEST_END)
        result = metrics(r[TARGET],r.predicted,r[TIME].to_numpy(),r.score_home)
        label = f'Tuned {config.minutes}-minute NN'
        rows.append(dict(model=label,config_id=int(cid),minutes=config.minutes,selected=int(int(cid)==selection['winner']),**result,**policy))
        predictions.append(r.assign(model=label))
        event,false = event_details(r[TARGET],r.predicted,r[TIME].to_numpy())
        pd.DataFrame(event).to_csv(out/f'events_{config.minutes}min.csv',index=False)
        pd.DataFrame(false).to_csv(out/f'false_events_{config.minutes}min.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'historical_metrics.csv',index=False)
    pd.concat(predictions).to_csv(out/'historical_predictions.csv',index=False)
    run_baselines(data,out,include_holdout=True)
    dump(out/'evaluation_receipt.json',dict(frozen_models_sha256=sha(out/'frozen_models.json'),
        selection_sha256=sha(out/'locked_selection.json'),historical_metrics_sha256=sha(out/'historical_metrics.csv'),
        selection_changed=False, fresh_unseen_test=False))
    print(pd.DataFrame(rows)[['model','selected','home_recall','away_recall','event_recall','event_precision','false_events_per_day']].to_string(index=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['search','baselines','prepare','evaluate','predict','smoke'])
    parser.add_argument('--data',type=Path,default=HERE/'data'/'occupancy.csv')
    parser.add_argument('--out',type=Path,default=HERE/'results')
    parser.add_argument('--model',type=Path)
    args = parser.parse_args()
    raw = load_data(args.data,labels=args.stage!='predict')
    dev = raw.loc[raw[TIME] < DEV_END].copy()
    if args.stage in ('search','smoke'):
        run_search(dev,args.out,smoke=args.stage=='smoke')
    elif args.stage == 'baselines':
        run_baselines(dev,args.out)
    elif args.stage == 'prepare':
        prepare_final(dev,args.out)
    elif args.stage == 'evaluate':
        evaluate_final(raw,args.out)
    else:
        if args.model is None:
            parser.error('--model is required for prediction')
        result = predict_saved(raw,args.model)
        args.out.mkdir(parents=True,exist_ok=True)
        result.to_csv(args.out/'new_predictions.csv',index=False)
        print(f'Wrote {len(result)} predictions to {args.out / "new_predictions.csv"}')


if __name__ == '__main__':
    main()
