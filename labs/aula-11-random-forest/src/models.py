"""Modelos, métricas e avaliação agrupada por participante."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from .data import STATIC_FEATURES, HISTORY_FEATURES, TARGET


def features_for(scenario: str) -> list[str]:
    if scenario not in ('contexto', 'historico'):
        raise ValueError("Use scenario='contexto' ou 'historico'.")
    return STATIC_FEATURES + (HISTORY_FEATURES if scenario == 'historico' else [])


def participant_split(df: pd.DataFrame, seed: int = 42, test_size: float = .25):
    a,b = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=seed).split(df, groups=df['subject']))
    assert not (set(df.iloc[a].subject) & set(df.iloc[b].subject))
    return np.asarray(a), np.asarray(b)


def make_pipeline(model_name: str, feature_names: list[str], seed: int = 42, quick: bool = False):
    categoricals = [c for c in ('condition','order') if c in feature_names]
    numerics = [c for c in feature_names if c not in categoricals]
    preprocessor = ColumnTransformer([
        ('numerico', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True))]), numerics),
        ('categorico', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categoricals),
    ], remainder='drop', sparse_threshold=0)
    if model_name == 'Decision Tree':
        reg = DecisionTreeRegressor(max_depth=6,min_samples_leaf=25,random_state=seed)
    elif model_name == 'Random Forest':
        reg = RandomForestRegressor(n_estimators=60 if quick else 180,
                                    max_depth=12,min_samples_leaf=6,
                                    max_features='sqrt',n_jobs=2,random_state=seed)
    elif model_name == 'XGBoost':
        reg = XGBRegressor(n_estimators=75 if quick else 180,max_depth=4,
                           learning_rate=.055,subsample=.85,colsample_bytree=.90,
                           reg_lambda=2.0,objective='reg:squarederror',
                           tree_method='hist',n_jobs=2,random_state=seed)
    else:
        raise ValueError(f'Modelo não suportado: {model_name}')
    return Pipeline([('preprocess',preprocessor),('model',reg)])


def scores(y, yhat) -> dict:
    yy=np.asarray(y,dtype=float);pp=np.asarray(yhat,dtype=float)
    return {'MAE':float(mean_absolute_error(yy,pp)),
            'RMSE':float(np.sqrt(mean_squared_error(yy,pp))),
            'R2':float(r2_score(yy,pp)),
            'bias_residual':float(np.mean(pp-yy))}


def participant_ci_for_delta(test_frame: pd.DataFrame, pred_a, pred_b,
                             n_boot: int = 2000, seed: int = 42) -> dict:
    """Intervalo bootstrap por PARTICIPANTE para ganho MAE(A)-MAE(B).

    Valores positivos: pred_b apresenta menor MAE. Resultados exploratórios,
    não equivalem a um teste causal ou a múltiplas divisões de treinamento.
    """
    r=test_frame[['subject',TARGET]].copy()
    yy=r[TARGET].to_numpy()
    r['err_a']=np.abs(yy-np.asarray(pred_a))
    r['err_b']=np.abs(yy-np.asarray(pred_b))
    unit=r.groupby('subject')[['err_a','err_b']].mean()
    diff=(unit.err_a-unit.err_b).to_numpy()
    rng=np.random.default_rng(seed)
    picks=rng.integers(0,len(diff),size=(n_boot,len(diff)))
    boots=diff[picks].mean(axis=1)
    return {'delta_mae_positive_favors_history':float(diff.mean()),
            'ci95_low':float(np.quantile(boots,.025)),
            'ci95_high':float(np.quantile(boots,.975)),
            'participants':int(len(diff))}


def group_cv_mae(model_name: str, scenario: str, train: pd.DataFrame,
                 seed: int=42, quick:bool=False, folds:int=4) -> dict:
    cols=features_for(scenario)
    splitter=GroupKFold(n_splits=folds)
    vals=[]
    for fit_idx,val_idx in splitter.split(train[cols],train[TARGET],groups=train.subject):
        est=make_pipeline(model_name,cols,seed,quick)
        est.fit(train.iloc[fit_idx][cols],train.iloc[fit_idx][TARGET])
        yp=est.predict(train.iloc[val_idx][cols])
        vals.append(float(mean_absolute_error(train.iloc[val_idx][TARGET],yp)))
    return {'cv_mae_mean':float(np.mean(vals)),
            'cv_mae_sd':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0,
            'folds':len(vals)}
