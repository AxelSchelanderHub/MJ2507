"""Editable JSON configuration: no model-search values live in the runner."""
from itertools import product
import math
import model_core as c

def candidates(cfg):
    search=cfg['search']
    if search['mode']=='recipes':
        return [c.Config(**{**cfg['model'],**r,'family':f})
                for f in search['families'] for r in search['recipes']]
    grid=search['grid']
    return [c.Config(**{**cfg['model'],**dict(zip(grid,values))})
            for values in product(*grid.values())]

def make_splits(cfg):
    s=cfg['split']
    kwargs={k:s[k] for k in ('stop_window_hours','gap_hours','calibration_hours')}
    folds=[c.temporal_split(start,c.pd.Timestamp(start)+c.pd.Timedelta(hours=s['score_hours']),i+1,**kwargs)
           for i,start in enumerate(s['validation_starts'])]
    final=c.temporal_split(s['test_start'],s['test_end'],len(folds)+1,**kwargs)
    return folds,final

def validate(cfg):
    expected={'model','search','split','thresholds','search_seeds','ensemble_seeds','finalist_count'}
    if set(cfg)!=expected: raise ValueError(f'Configuration keys must be {sorted(expected)}')
    required=set(c.Config.__dataclass_fields__)
    if set(cfg['model'])!=required: raise ValueError('model must explicitly specify every Config field; see config.json')
    if cfg['search']['mode'] not in ('recipes','grid'): raise ValueError('search mode must be recipes or grid')
    for key in ('search_seeds','ensemble_seeds'):
        if not cfg[key] or any(type(v)!=int or v<0 for v in cfg[key]): raise ValueError(f'Invalid {key}')
    if type(cfg['finalist_count'])!=int or cfg['finalist_count']<1: raise ValueError('finalist_count must be positive')
    if not cfg['thresholds'] or any(not math.isfinite(t) or not 0<t<1 for t in cfg['thresholds']):
        raise ValueError('thresholds must contain finite probabilities strictly between 0 and 1')
    s=cfg['split']
    for key in ('stop_window_hours','calibration_hours','score_hours'):
        if not math.isfinite(s[key]) or s[key]<=0: raise ValueError(f'{key} must be positive')
    folds,final=make_splits(cfg)
    if not folds or final['score_end']<=final['score_start']: raise ValueError('Invalid scoring dates')
    if any(f['score_end']>final['score_start'] for f in folds): raise ValueError('Development scoring must finish before test starts')
    models=[c.Config(**cfg['model']),*candidates(cfg)]
    if len(models)==1: raise ValueError('Search cannot be empty')
    for m in models:
        if m.minutes!=5: raise ValueError('This package is specifically a five-minute model')
        if m.family not in ('levels','levels_clock','dynamics','dynamics_clock'): raise ValueError('Invalid family')
        if m.activation not in ('tanh','relu') or m.optimizer not in ('adam','gd'): raise ValueError('Invalid activation/optimizer')
        for k in ('hidden','epochs','check_every','patience_checks','history_minutes'):
            if type(getattr(m,k))!=int or getattr(m,k)<1: raise ValueError(f'{k} must be a positive integer')
        for k in ('lr','adam_epsilon','init_scale','scale_floor','adam_gradient_weight1','adam_gradient_weight2'):
            if not math.isfinite(getattr(m,k)) or getattr(m,k)<=0: raise ValueError(f'{k} must be positive')
        for k in ('l2','class_power','min_delta','transition_minutes','transition_max_gap_minutes'):
            if not math.isfinite(getattr(m,k)) or getattr(m,k)<0: raise ValueError(f'{k} cannot be negative')
        if not math.isfinite(m.transition_boost) or m.transition_boost<1: raise ValueError('transition_boost must be >= 1')
        if not all(0<=v<1 for v in (m.adam_beta1,m.adam_beta2)): raise ValueError('Adam betas must be in [0,1)')
        if not 0<=m.night_end_hour<m.night_start_hour<=24: raise ValueError('Night interval must cross midnight')
        if not m.delta_minutes or any(type(v)!=int or v<5 or v%5 for v in m.delta_minutes): raise ValueError('delta_minutes must be positive multiples of 5')
