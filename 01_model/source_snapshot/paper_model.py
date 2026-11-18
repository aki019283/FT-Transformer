"""Load unchanged architecture AST from the paper authors' archived source."""
import ast
import hashlib
import json
import math
import typing as ty
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F, init as nn_init

ROOT = Path(__file__).parent

def load_official():
    provenance = json.loads((ROOT/'upstream.json').read_text(encoding='utf8'))
    for name, digest in provenance['sha256'].items():
        if hashlib.sha256((ROOT/'vendor'/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Official source hash mismatch: '+name)
    env = dict(math=math, ty=ty, np=np, torch=torch, nn=nn, F=F, nn_init=nn_init, Tensor=Tensor)
    for filename, names in [
        ('lib__deep.py', ['reglu','geglu','get_activation_fn','get_nonglu_activation_fn']),
        ('bin__ft_transformer.py', ['Tokenizer','MultiheadAttention','Transformer'])]:
        tree=ast.parse((ROOT/'vendor'/filename).read_text(encoding='utf8'))
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        if len(nodes)!=len(names): raise ValueError('Unexpected upstream structure')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,'exec'),env)
        if filename=='lib__deep.py':
            env['lib']=SimpleNamespace(**{n:env[n] for n in names})
    return env['Transformer']

OfficialTransformer=load_official()

def audit_raw_features(frame, features, destination, allow_rank64=False):
    """Audit before project sanitize can cast overflow to inf and erase it."""
    records=[]
    for column in features:
        values=frame[column].to_numpy(dtype=np.float64)
        bad=np.isinf(values) | (np.isfinite(values) & (np.abs(values)>np.finfo(np.float32).max))
        ix=np.flatnonzero(bad)
        samples=[]
        for i in ix[:5]:
            row=frame.iloc[int(i)]
            samples.append(dict(TradingDate=str(row.get('TradingDate','')),Stkcd=str(row.get('Stkcd','')),value=str(values[i])))
        records.append(dict(feature=column,rows=len(values),missing=int(np.isnan(values).sum()),
            infinite=int(np.isinf(values).sum()),finite_float32_overflow=int((np.isfinite(values)&bad).sum()),samples=samples))
    failed=any(r['infinite'] or (r['finite_float32_overflow'] and not allow_rank64) for r in records)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(dict(state='BLOCKED_RAW_NUMERIC' if failed else 'PASS',finite_large_value_policy='Rank in float64 before float32 conversion' if allow_rank64 else 'Block',columns=records),ensure_ascii=False,indent=2),encoding='utf8')
    if failed:raise ValueError('Raw factor infinity/float32 overflow BEFORE sanitize. No values replaced. Read '+str(destination))

def rank64_frame(frame, features, target, time_col, transform):
    expected=[{'kind':'sanitize'},{'kind':'cross_section_rank','params':{'output_range':[-1.0,1.0],'rank_target':True}}]
    if transform.get('dataset')!=expected:
        raise ValueError('rank64 adapter requires the exact registered sanitize/rank config')
    result=frame.copy()
    for column in [*features,target]:
        values=frame[column].to_numpy(dtype=np.float64)
        if np.isinf(values).any():raise ValueError('Infinite raw value in '+column)
    # Rank finite raw magnitudes without clipping or narrowing to float32 first.
    # Preserve original average-tie ranks, missing values, target ranks, and scaling order.
    ranks=frame.groupby(time_col,sort=False)[features].rank(pct=True,method='average')
    values=ranks.to_numpy(dtype=np.float32)
    result[features]=-1.0+values*2.0
    targets=frame.groupby(time_col,sort=False)[target].rank(pct=True,method='average')
    result[target]=-1.0+targets.to_numpy(dtype=np.float32)*2.0
    return result

class PaperFT(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.core=OfficialTransformer(d_numerical=features,categories=None,token_bias=True,
            n_layers=3,d_token=192,n_heads=8,d_ffn_factor=4/3,
            attention_dropout=.2,ffn_dropout=.1,residual_dropout=0.,
            activation='reglu',prenormalization=True,initialization='kaiming',
            kv_compression=None,kv_compression_sharing=None,d_out=1)
    def forward(self,x):
        return self.core(x,None)

def optimizer_groups(core):
    # Exact official name-based decay rule (names must be relative to the core).
    needs=lambda n: all(x not in n for x in ['tokenizer','.norm','.bias'])
    return [{'params':[p for n,p in core.named_parameters() if needs(n)]},
            {'params':[p for n,p in core.named_parameters() if not needs(n)],'weight_decay':0.}]

def safe_features(frame):
    a=frame.to_numpy(dtype=np.float64)
    if np.isinf(a).any() or (np.abs(a[np.isfinite(a)])>np.finfo(np.float32).max).any():
        raise ValueError('Infinite/float32-overflow features: audit source and transforms before training')
    # Original pipeline explicitly specifies constant-zero missing-value imputation.
    return np.nan_to_num(a,nan=0.).astype(np.float32)
