from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error
from src.data import (load_data, audit_data, make_features, STATIC_FEATURES,
                      HISTORY_FEATURES, TARGET, file_sha256)
from src.models import (participant_split, features_for, make_pipeline,
                        participant_ci_for_delta, scores)

ROOT=Path(__file__).resolve().parents[1]
CSV=ROOT/'data'/'Exp2.csv'

@pytest.fixture(scope='module')
def dataset():
    return load_data(CSV)

@pytest.fixture(scope='module')
def frame(dataset):
    return make_features(dataset)


def test_source_hash(dataset):
    assert file_sha256(CSV)=='ec2207efc6fa936031107376857d48310990874d200de39c2880945ace574277'
    assert dataset.shape==(14400,11)


def test_audit_strict(dataset):
    out=audit_data(dataset, strict=True)
    assert out['participants']==120
    assert out['condition_counts']=={'0':3600,'1':3600,'2':3600,'3':3600}
    assert out['missing_by_column']['collaboration']==3600
    assert out['duplicated_exact_rows']==1
    assert len(out['sequence_patterns'])==3


def test_trials_and_block_boundaries(frame):
    for _,g in frame.groupby('subject'):
        assert g.trial_index.tolist()==list(range(1,121))
        assert g.block_index.tolist()==[0]*30+[1]*30+[2]*30+[3]*30
        assert g.trial_in_block.tolist()==list(range(1,31))*4
        assert g.iloc[:30]['shown_ai_bias'].isna().all()


def test_features_lagged_and_no_present_information(dataset):
    # Reescrever o trial 35 não deve alterar os preditores do trial 35.
    original=make_features(dataset)
    d=dataset.copy()
    i=d.index[(d.subject==1)].tolist()[34]
    d.loc[i,'response']=91.0
    d.loc[i,'confidence']=0.123
    d.loc[i,'collaboration']=0.678
    d.loc[i,'responseAI']=11.0
    changed=make_features(d)
    cols=features_for('historico')
    pd.testing.assert_series_equal(original.loc[i,cols],changed.loc[i,cols])
    # Mas alvos e lags da próxima tentativa devem reagir à mudança.
    assert original.loc[i,TARGET]!=changed.loc[i,TARGET]
    assert original.loc[i+1,'lag1_error']!=changed.loc[i+1,'lag1_error']
    assert original.loc[i+1,'lag1_shown_ai_bias']!=changed.loc[i+1,'lag1_shown_ai_bias']


def test_no_inter_subject_leakage(frame):
    for _,g in frame.groupby('subject',sort=False):
        first=g.iloc[0]
        assert np.isnan(first['lag1_error'])
        assert np.isnan(first['rolling5_prior_error'])
        assert np.isnan(first['lag1_shown_ai_bias'])


def test_forbidden_columns_not_selected():
    forbidden={'subject','response','responseAI','confidence','collaboration',
               'perceivedInfluenceA','perceivedInfluenceB','perceivedInfluenceN',TARGET}
    assert not (set(features_for('historico'))&forbidden)
    assert len(features_for('historico'))>len(features_for('contexto'))


def test_split_no_same_participant(frame):
    a,b=participant_split(frame,seed=42)
    assert len(set(frame.iloc[a].subject))==90
    assert len(set(frame.iloc[b].subject))==30
    assert set(frame.iloc[a].subject).isdisjoint(set(frame.iloc[b].subject))


@pytest.mark.parametrize('name',['Decision Tree','Random Forest','XGBoost'])
def test_each_estimator_works(frame,name):
    # Duas dezenas de participantes bastam para teste de integração rápido.
    small=frame[frame.subject<=12].copy()
    a,b=participant_split(small,test_size=0.25,seed=12)
    col=features_for('historico')
    est=make_pipeline(name,col,seed=12,quick=True)
    est.fit(small.iloc[a][col],small.iloc[a][TARGET])
    yp=est.predict(small.iloc[b][col])
    assert len(yp)==len(b)
    assert np.isfinite(yp).all()
    assert mean_absolute_error(small.iloc[b][TARGET],yp)>=0


def test_bootstrap_interval_on_groups(frame):
    f=frame[frame.subject<=5].copy()
    y=f[TARGET].values
    p1=np.zeros(len(y))
    p2=np.zeros(len(y))
    out=participant_ci_for_delta(f,p1,p2,n_boot=100,seed=77)
    assert out['delta_mae_positive_favors_history']==0.0
    assert out['ci95_low']==0.0==out['ci95_high']
    assert out['participants']==5


def test_invalid_scenario():
    with pytest.raises(ValueError):
        features_for('futuro')
