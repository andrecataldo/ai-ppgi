"""Contrato do Exp2 e engenharia de atributos estritamente retrospectivos.

Todas as linhas são mantidas na ordem original do CSV. Nenhum atributo da
resposta/IA/confiança/ponderação da tentativa corrente pode entrar no modelo.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ORIGINAL_COLUMNS = [
    'subject', 'evidence', 'response', 'responseAI', 'confidence',
    'collaboration', 'condition', 'order', 'perceivedInfluenceA',
    'perceivedInfluenceB', 'perceivedInfluenceN',
]
CONDITIONS = {0: 'Sem IA', 1: 'IA precisa', 2: 'IA enviesada', 3: 'IA ruidosa'}
# Conjunto de atributos definidos ANTES da resposta na tentativa t.
STATIC_FEATURES = [
    'evidence', 'condition', 'order', 'trial_index', 'trial_in_block',
]
HISTORY_FEATURES = [
    'lag1_error', 'rolling5_prior_error', 'prior_mean_error',
    'rolling5_prior_absolute_error', 'lag1_confidence',
    'rolling5_prior_confidence', 'lag1_shown_ai_bias',
    'rolling5_prior_shown_ai_bias', 'prior_mean_shown_ai_bias',
    'rolling5_prior_collaboration',
]
TARGET = 'signed_error'


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path: str | Path) -> pd.DataFrame:
    """Carrega sem ordenar: a ordem das linhas no arquivo marca o tempo."""
    frame = pd.read_csv(path)
    missing = set(ORIGINAL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'Colunas obrigatórias ausentes: {sorted(missing)}')
    return frame.loc[:, ORIGINAL_COLUMNS].copy()


def audit_data(df: pd.DataFrame, strict: bool = True) -> dict:
    if df.empty:
        raise ValueError('O dataset está vazio.')
    if df['subject'].isna().any() or df['condition'].isna().any():
        raise ValueError('Participante e condição são obrigatórios.')
    if not set(df['condition'].unique()).issubset(CONDITIONS):
        raise ValueError('Há códigos de condição não esperados.')
    if df[['subject','evidence','response','responseAI','confidence','order']].isna().any().any():
        raise ValueError('Ausências em colunas estruturais ou da tarefa.')
    missing_collab = df['collaboration'].isna()
    if not (missing_collab == (df['condition'] == 0)).all():
        raise ValueError('A ausência em collaboration deve ocorrer exatamente no baseline.')
    per_subject = df.groupby('subject', sort=False).size()
    sequences = {}
    for pid, s in df.groupby('subject', sort=False):
        cond = s['condition'].to_numpy()
        seq = [(int(cond[0]),1)]
        for value in cond[1:]:
            value = int(value)
            if value == seq[-1][0]:
                seq[-1] = (value, seq[-1][1]+1)
            else:
                seq.append((value,1))
        sequences[int(pid)] = tuple(seq)
        if strict and (len(seq) != 4 or seq[0] != (0,30)
                       or sorted(c for c,_ in seq) != [0,1,2,3]
                       or any(n != 30 for _,n in seq)):
            raise ValueError(f'Sequência temporal inesperada no participante {pid}: {seq}')
    if strict and (len(per_subject) != 120 or not (per_subject == 120).all()
                   or len(df) != 14400):
        raise ValueError('Exp2.csv diferente do baseline original (120x120).')
    if not (df['evidence'].between(0,100) & df['response'].between(0,100)
            & df['responseAI'].between(0,100) & df['confidence'].between(0,1)).all():
        raise ValueError('Valores fora da faixa física esperada.')
    summary = {
        'rows': int(len(df)), 'participants': int(df.subject.nunique()),
        'columns': int(len(df.columns)),
        'trials_per_participant': {'min':int(per_subject.min()),'max':int(per_subject.max())},
        'condition_counts':{str(k):int(v) for k,v in df.condition.value_counts().sort_index().items()},
        'missing_by_column':{k:int(v) for k,v in df.isna().sum().items()},
        'duplicated_exact_rows':int(df.duplicated().sum()),
        'order_subject_counts':{str(k):int(v) for k,v in df.groupby('subject').order.first().value_counts().sort_index().items()},
        'sequence_patterns': {str(x): sum(1 for y in sequences.values() if y == x) for x in sorted(set(sequences.values()))},
        'strict_validation':bool(strict),
    }
    return summary


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Toda feature history(t) depende apenas de linhas com tempo < t.

    Não reordene: o índice do arquivo representa a sequência de tentativas
    de cada participante. Geração do contexto 'evidence' assume a perspectiva
    do pesquisador, que conhece o estímulo antes da estimativa humana.
    """
    out = df.reset_index(drop=True).copy()
    out['trial_index'] = out.groupby('subject', sort=False).cumcount() + 1
    out['block_index'] = (out['trial_index'] - 1) // 30  # 0 baseline; 1-3 interação
    out['trial_in_block'] = (out['trial_index'] - 1) % 30 + 1
    out[TARGET] = out['response'] - out['evidence']
    out['absolute_error'] = out[TARGET].abs()
    # A resposta da IA somente é mostrada no fim da tentativa de interação.
    # responseAI do baseline não representa exposição a aconselhamento.
    out['shown_ai_bias'] = (out['responseAI'] - out['evidence']).where(out['condition'] != 0)
    g = out.groupby('subject', sort=False)
    out['lag1_error'] = g[TARGET].shift(1)
    out['rolling5_prior_error'] = g[TARGET].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    out['prior_mean_error'] = g[TARGET].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean())
    out['rolling5_prior_absolute_error'] = g['absolute_error'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    out['lag1_confidence'] = g['confidence'].shift(1)
    out['rolling5_prior_confidence'] = g['confidence'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    out['lag1_shown_ai_bias'] = g['shown_ai_bias'].shift(1)
    out['rolling5_prior_shown_ai_bias'] = g['shown_ai_bias'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    out['prior_mean_shown_ai_bias'] = g['shown_ai_bias'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean())
    out['rolling5_prior_collaboration'] = g['collaboration'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    return out


def descriptive_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if TARGET not in work:
        work[TARGET] = work['response'] - work['evidence']
        work['absolute_error'] = work[TARGET].abs()
    return work.groupby(['subject','condition'], as_index=False).agg(
        bias_mean=(TARGET,'mean'), mae=('absolute_error','mean'), trials=(TARGET,'size'))
