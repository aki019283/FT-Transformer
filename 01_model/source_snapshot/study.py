"""Real-data rolling comparison. Never substitutes simulated performance for failures."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
import yaml
from paper_model import PaperFT, optimizer_groups, safe_features, audit_raw_features, rank64_frame
from registration import register
from standard_sharpe import standard_sharpe, CONVENTION
from holding_month import align_dataset, check_label_availability, audit_feature_timing, relabel_daily_derived_months, POLICY

KEYS = ['TradingDate', 'Stkcd']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, p)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def predict(net, x, device, batch):
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(torch.as_tensor(x[i:i+batch], device=device)).cpu().numpy()
                               for i in range(0, len(x), batch)])


def fit(net, tr, va, features, target, cfg, update, seed, device, batch):
    from training.batching import BuildPeriodBatches
    seed_all(seed)
    excluded_train=int((~np.isfinite(tr[target])).sum());excluded_valid=int((~np.isfinite(va[target])).sum())
    tr=tr.loc[np.isfinite(tr[target])].reset_index(drop=True)
    va=va.loc[np.isfinite(va[target])].reset_index(drop=True)
    def arrays(d):
        y = d[target].to_numpy(np.float32)
        if not np.isfinite(y).all(): raise ValueError('Nonfinite train/validation labels')
        return safe_features(d[features]), y
    x,y = arrays(tr); vx,vy = arrays(va)
    if not len(y) or not len(vy): raise ValueError('Empty train/validation set')
    lr = cfg['optimizer']['learning_rate'] * (cfg.get('update_learning_rate_scale', .25) if update else 1)
    opt = (torch.optim.AdamW(optimizer_groups(net.core), lr=1e-4, weight_decay=1e-5) if isinstance(net,PaperFT) else torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=cfg['optimizer']['weight_decay']))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=cfg.get('lr_scheduler_patience',4),
        factor=cfg.get('lr_scheduler_factor',.5), min_lr=cfg.get('min_learning_rate',1e-5))
    best, stale, history, state = math.inf, 0, [], None
    for epoch in range(cfg['update_epochs'] if update else cfg['full_epochs']):
        net.train(); losses=[]
        for ix in BuildPeriodBatches(tr.TradingDate.to_numpy(), cfg.get('periods_per_batch',8)):
            opt.zero_grad(set_to_none=True); total=0.
            # Preserve an optimizer step per original period-batch. Microbatches avoid attention OOM.
            for j in range(0,len(ix),batch):
                sub=ix[j:j+batch]
                pred=net(torch.as_tensor(x[sub],device=device))
                loss=torch.nn.functional.mse_loss(pred,torch.as_tensor(y[sub],device=device))
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                (loss * (len(sub)/len(ix))).backward(); total+=float(loss.detach())*len(sub)/len(ix)
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step(); losses.append(total)
        score=float(np.mean((predict(net,vx,device,batch)-vy)**2)); sched.step(score)
        history.append(dict(epoch=epoch+1,train_mse=float(np.mean(losses)),valid_mse=score))
        if np.isfinite(score) and score < best-cfg.get('early_stopping_min_delta',1e-6):
            best=score; state=copy.deepcopy(net.state_dict()); best_epoch=epoch+1; stale=0
        else: stale+=1
        if stale>=cfg.get('early_stopping_patience',12): break
    if state is None: raise ValueError('No finite validation checkpoint')
    net.load_state_dict(state)
    return dict(best_epoch=best_epoch,validation_mse=best,history=history,train_rows=len(y),valid_rows=len(vy),excluded_missing_train_labels=excluded_train,excluded_missing_valid_labels=excluded_valid)


def block_interval(delta, block, repetitions=2000):
    """Resample observed paired month differences; never invent market observations."""
    a=np.asarray(delta,float)
    if len(a)<block*2 or not np.isfinite(a).all(): return [None,None]
    rng=np.random.default_rng(814); means=[]
    for _ in range(repetitions):
        starts=rng.integers(0,len(a)-block+1,size=math.ceil(len(a)/block))
        sample=np.concatenate([a[s:s+block] for s in starts])[:len(a)]
        means.append(sample.mean())
    return np.quantile(means,[.025,.975]).tolist()


def evaluate(joined, raw, target, out):
    labels=raw.set_index(KEYS)[target].reindex(joined.index)
    rows=[]
    for date,g in joined.groupby(level=0):
        y=labels.reindex(g.index)
        for model in joined:
            s=g[model]; k=max(1,math.ceil(.1*len(s)))
            observed=np.isfinite(y);top=y.loc[s.nlargest(k).index]
            if observed.sum()<3:raise ValueError('Insufficient observed labels for predictive evaluation')
            rows.append(dict(date=date,holding_month=str(pd.Timestamp(date).to_period('M')+1),model=model,n=len(s),observed_labels=int(observed.sum()),missing_labels=int((~observed).sum()),rank_ic=s[observed].corr(y[observed],method='spearman'),
                ic=s[observed].corr(y[observed]),top_label=top.mean() if np.isfinite(top).all() else np.nan,universe_label=y[observed].mean()))
    monthly=pd.DataFrame(rows)
    if not np.isfinite(monthly[['ic','rank_ic']]).all().all(): raise ValueError('Undefined predictive metric')
    monthly.to_csv(out/'report/monthly.csv',index=False,encoding='utf-8-sig')
    summary=monthly.groupby('model').agg(months=('date','count'),mean_ic=('ic','mean'),mean_rank_ic=('rank_ic','mean'),
        top_label_observed_months=('top_label','count'),missing_labels=('missing_labels','sum'),std_rank_ic=('rank_ic','std'),positive_month_fraction=('rank_ic',lambda x:(x>0).mean()),top_label=('top_label','mean'))
    summary['rank_icir']=summary.mean_rank_ic/summary.std_rank_ic
    summary.to_csv(out/'report/summary.csv',encoding='utf-8-sig')
    monthly.assign(year=pd.PeriodIndex(monthly.holding_month,freq='M').year).groupby(['year','model']).agg(months=('rank_ic','count'),mean_rank_ic=('rank_ic','mean')).to_csv(out/'report/annual.csv')
    pivot=monthly.pivot(index='date',columns='model',values='rank_ic'); evidence=[]
    for model in pivot:
        if model=='mlp_ensemble': continue
        delta=pivot[model]-pivot.mlp_ensemble
        evidence.append(dict(model=model,baseline='mlp_ensemble',paired_mean_delta=delta.mean(),wins=int((delta>0).sum()),
            months=len(delta),block3_ci95=block_interval(delta,3),block6_ci95=block_interval(delta,6)))
    save(out/'report/reliability.json',dict(comparisons=evidence,
        status='Historical exploratory evidence, NOT untouched holdout or proof of investability',
        limitations=['Repeatedly inspected history; no statistically independent final holdout',
        'Overlapping rolling windows and warm starts induce dependence; block intervals are approximate, unadjusted for multiple candidates',
        'Upstream point-in-time feature availability not verified; check reporting lags, survivorship and corporate actions',
        'Top label return is not an executed or cost-adjusted return',
        'High Sharpe requires complete holdings pricing, cost stress, drawdown review, and future frozen-model monitoring']))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--project-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',default='2021-01-01',help='First holding month');p.add_argument('--end',default='2026-08-31',help='Last holding month; default latest mature month')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,123,2026]);p.add_argument('--batch',type=int,default=128)
    p.add_argument('--smoke',action='store_true'); args=p.parse_args()
    register(args, Path(__file__).parent)
    alignment_path=Path(__file__).parent/'date_alignment_status.json'
    provenance_path=Path(__file__).parent/'factor_provenance_status.json'
    if not alignment_path.exists() or json.loads(alignment_path.read_text(encoding='utf-8'))['state']!='REBUILD_FROM_PRICES_V3_SOURCE_MONTH_PLUS_ONE':
        raise ValueError('Expected independently tested holding-month label rebuild package V3')
    if not provenance_path.exists() or json.loads(provenance_path.read_text(encoding='utf-8'))['state']!='DAILY_LAST_SAME_MONTH_RELABELED_PLUS_ONE_V1':
        raise ValueError('Expected verified daily-to-monthly factor provenance package')
    root=args.project_root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    for sub in ['report','factors','windows']: (out/sub).mkdir(exist_ok=True)
    sys.path.insert(0,str(root/'src'));os.chdir(root)
    from config import MlpFactorConfig
    from data import PanelDataset
    from transforms import BuildDatasetPipeline
    from windows import WindowBuilder
    from models.mlp import _ResidualMlpNetwork
    cp=root/'config/mlp_warm/old(58)+history-rcykc-multi(45).yaml';cfg=MlpFactorConfig.Load(cp)
    dc=cfg.Section('data');tc=cfg.Section('training');transform=cfg.Section('transforms')
    if transform.get('window') != [{'kind':'impute','params':{'strategy':'constant','value':0}}]: raise ValueError('Unsupported window transform')
    if tc['policy']!='warm_start' or tc['loss']['kind']!='mse': raise ValueError('Expected original warm-start MSE baseline')
    ds=PanelDataset.Load(cfg.DataPath(),time_col=dc['time_col'],asset_col=dc['asset_col'],target_col=dc['target_col'],feature_cols=dc.get('feature_cols'))
    features=list(ds.feature_cols)
    audit_raw_features(ds.ToFrame(),features,out/'report/raw_numeric_audit.json',allow_rank64=True)
    alignment_inputs=[Path(cfg.DataPath()),root/'data/source_data/adj_close.parquet',root/'data/source_data/Calendar.parquet']
    alignment_hashes={str(f):sha(f) for f in alignment_inputs}
    holding_start=pd.Timestamp(args.start).to_period('M')
    source_frame=relabel_daily_derived_months(ds.ToFrame(),dc['target_col'],out/'report')
    raw,schedule,price_cutoff=align_dataset(source_frame,root/'data/source_data',dc['target_col'],out/'report')
    if any(sha(Path(f))!=h for f,h in alignment_hashes.items()):raise ValueError('Inputs changed during label reconstruction')
    holding_end=pd.Timestamp(args.end).to_period('M') if args.end else pd.Period(schedule.holding_month.max(),freq='M')
    if str(holding_end) not in set(schedule.holding_month):raise ValueError('Requested holding month has no observed exit prices')
    args.start=str((holding_start-1).start_time.date());args.end=str((holding_end-1).end_time.date())
    save(out/'report/date_policy.json',dict(policy=POLICY,holding_start=str(holding_start),holding_end=str(holding_end),price_cutoff=str(price_cutoff),original_target_used=False,source_month_relabel='+1 month for all 103 daily-derived Ashare_new and rcykc_multi features',missing_training_labels='Excluded explicitly with counts; test predictions retain all stocks, missing held prices still block trading',historical_point_in_time_availability='Daily-to-monthly generator inspected: source month used its last daily observation, so source rows are relabeled to the following holding month. Individual vendor release lags beyond this price-data rule remain unaudited.'))
    timing_review=audit_feature_timing(raw,features,dc['target_col'],args.start,args.end,out/'report')
    save(out/'report/feature_timing_review.json',timing_review)
    if timing_review['review_required']:print('WARNING: feature timing requires source-level review; do not treat high IC/Sharpe as validated alpha.',flush=True)
    ds=PanelDataset.FromFrame(raw,dc['time_col'],dc['asset_col'],dc['target_col'],features)
    raw=ds.ToFrame().copy();frame=rank64_frame(raw,features,dc['target_col'],dc['time_col'],transform)
    for d in [raw,frame]:
        d.TradingDate=pd.to_datetime(d.TradingDate)
        if d.duplicated(KEYS).any():raise ValueError('Duplicate raw keys')
    wc=cfg.Section('window');wc.update(test_start=args.start,test_end=args.end)
    windows=WindowBuilder(wc).Build(PanelDataset.FromFrame(frame,dc['time_col'],dc['asset_col'],dc['target_col'],features))
    if args.smoke: windows=windows[:2];tc=dict(tc,full_epochs=2,update_epochs=2);args.seeds=[42]
    if not windows:raise ValueError('No windows')
    expected=pd.period_range(args.start,args.end,freq='M')
    if not args.smoke and [pd.Timestamp(w.test_periods[0]).to_period('M') for w in windows]!=list(expected):raise ValueError('Missing expected test months')
    for w in windows:
        if pd.Timestamp(max(w.valid_periods))+pd.offsets.MonthEnd(1)>=pd.Timestamp(min(w.test_periods)):raise ValueError('Insufficient validation/test purge')
        if pd.Timestamp(max(w.train_periods))+pd.offsets.MonthEnd(1)>=pd.Timestamp(min(w.valid_periods)):raise ValueError('Insufficient training/validation purge')
    check_label_availability(windows,schedule).to_csv(out/'report/label_availability.csv',index=False)
    price_path=root/'data/source_data/adj_close.parquet'
    dates=pd.read_parquet(price_path,columns=['TradingDate']).TradingDate
    if schedule.loc[schedule.holding_month.eq(str(holding_end)),'exit'].iloc[0]>dates.max():raise ValueError('Requested holding-period exit not observed')
    market_inputs=[price_path]+[root/'data/source_data'/f for f in ['Calendar.parquet','Suspension_Limit.parquet','history-new.pq','history-1800.pq','指数日行情.parquet']]
    originals={'original_M0':root/'data/experiments/nn_calendar_huber/calendar_v2/04_factors/seed42_training/M0.parquet',
               'original_E0':root/'data/experiments/nn_followup_validation/04_factors/E0_warm_mse_ensemble3.parquet'}
    hashes={str(f):sha(f) for f in [cp,Path(cfg.DataPath()),*market_inputs,*[v for v in originals.values() if v.exists()],
        *sorted((root/'src').rglob('*.py')),*sorted(Path(__file__).parent.rglob('*.py')),Path(__file__).parent/'protocol.json',Path(__file__).parent/'upstream.json',
        Path(__file__).parent/'backtest_template.yaml',alignment_path,provenance_path]}
    ident=dict(hashes=hashes,args=vars(args),features=features,training=tc,runtime={'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__},windows=[dict(train=list(w.train_periods),valid=list(w.valid_periods),test=list(w.test_periods)) for w in windows])
    # Normalize Paths to strings before equality checks.
    ident=json.loads(json.dumps(ident,default=str))
    identity=out/'identity.json'
    if identity.exists() and json.loads(identity.read_text(encoding='utf-8'))!=ident:raise ValueError('Resume identity mismatch; new output directory required')
    save(identity,ident);save(out/'report/status.json',dict(state='TRAINING',smoke=args.smoke))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device='cuda' if torch.cuda.is_available() else 'cpu';preds={};first=None
    for model in ['mlp','ft_transformer']:
        for seed in args.seeds:
            seed_all(seed)
            net=(_ResidualMlpNetwork(len(features),float(cfg.Section('model')['params']['dropout'])) if model=='mlp' else PaperFT(len(features))).to(device)
            name=f'{model}_s{seed}';parts=[]
            for k,w in enumerate(windows):
                dest=out/'windows'/name/w.window_id;dest.mkdir(parents=True,exist_ok=True)
                marker=dest/'COMMITTED.json'
                if marker.exists():
                    for f,h in json.loads(marker.read_text()).items():
                        if sha(dest/f)!=h:raise ValueError('Checkpoint hash mismatch')
                    net.load_state_dict(torch.load(dest/'model.pt',map_location=device,weights_only=True));parts.append(pd.read_parquet(dest/'prediction.parquet'));continue
                tr=frame.loc[frame.TradingDate.isin(w.train_periods)];va=frame.loc[frame.TradingDate.isin(w.valid_periods)];te=frame.loc[frame.TradingDate.isin(w.test_periods)]
                if te.empty:raise ValueError('Empty test window')
                started=time.monotonic();diag=fit(net,tr,va,features,dc['target_col'],tc,k>0,seed+k,device,args.batch)
                x=safe_features(te[features])
                pr=te[KEYS].copy();pr[name]=predict(net,x,device,args.batch)
                if not np.isfinite(pr[name]).all():raise ValueError('Nonfinite predictions')
                diag.update(parameters=sum(v.numel() for v in net.parameters()),seconds=time.monotonic()-started,model=model,seed=seed)
                save(dest/'diagnostics.json',diag);torch.save(net.state_dict(),dest/'model.pt');pr.to_parquet(dest/'prediction.parquet',index=False)
                save(marker,{f:sha(dest/f) for f in ['model.pt','prediction.parquet','diagnostics.json']});parts.append(pr)
                print(f'{name}: {k+1}/{len(windows)} windows complete',flush=True)
            full=pd.concat(parts).sort_values(KEYS).set_index(KEYS)
            if full.index.has_duplicates:raise ValueError('Duplicate predictions')
            if first is None:first=full.index
            if not first.equals(full.index):raise ValueError('Unmatched model keys')
            preds[name]=full[name]
    joined=pd.DataFrame(preds)
    for model in ['mlp','ft_transformer']:
        joined[model+'_ensemble']=joined[[f'{model}_s{s}' for s in args.seeds]].mean(axis=1)
    (out/'predictions_holding_month').mkdir(exist_ok=True)
    for name in joined:
        exported=joined[[name]].reset_index()
        exported.to_parquet(out/'factors'/f'{name}.parquet',index=False)
        exported.TradingDate=(exported.TradingDate.dt.to_period('M')+1).dt.to_timestamp('M')
        exported.to_parquet(out/'predictions_holding_month'/f'{name}.parquet',index=False)
    evaluate(joined,raw,dc['target_col'],out)
    if any(sha(Path(f))!=h for f,h in hashes.items()):raise ValueError('Inputs changed during study')
    # Archived predictions were trained with legacy next-month targets and timing.
    # Moving their dates earlier would not establish point-in-time availability.
    save(out/'report/original_baseline.json',dict(
        records=[dict(name=n,path=str(f),exists=f.exists(),state='EXCLUDED_LEGACY_TIMING') for n,f in originals.items()],
        note='Fair baseline is the original residual MLP architecture retrained on corrected labels, same windows and seeds. Legacy predictions are not shifted earlier or ranked against this experiment.'))
    bt=yaml.safe_load((Path(__file__).parent/'backtest_template.yaml').read_text(encoding='utf-8'))
    for key,val in bt['data']['source_data'].items():bt['data']['source_data'][key]=str(root/'data/source_data'/Path(val).name)
    bt['factors']=[str(out/'factors'/f'{m}_ensemble.parquet') for m in ['mlp','ft_transformer']]
    for n in originals:
        if n in joined:bt['factors'].append(str(out/'factors'/f'{n}.parquet'))
    bt['backtest'].update(period={'start_date':args.start,'end_date':args.end},direction='high',execution='next_close',benchmark_same_pool=True)
    # Fixed, observed index benchmark; avoids conflating incomplete pool bookkeeping with the model.
    bt['benchmark']=[{'mode':'index','index_code':'000852'}]
    bt['output']={'output_dir':str(out/'strict_backtest')};bt['log']={'log_file':str(out/'strict_backtest.log'),'level':'INFO'}
    (out/'backtest.yaml').write_text(yaml.safe_dump(bt,allow_unicode=True,sort_keys=False),encoding='utf-8')
    code=None
    if not args.smoke:
        with (out/'strict_console.log').open('w',encoding='utf-8') as log:
            code=subprocess.run([sys.executable,str(Path(__file__).parent/'strict_run.py'),'--config',str(out/'backtest.yaml')],stdout=log,stderr=subprocess.STDOUT).returncode
    cost_status={0:code}; trading=[]
    if code==0:
        for bps in [5,10]:
            stress=copy.deepcopy(bt);stress['output']['output_dir']=str(out/f'strict_cost{bps}')
            stress['log']['log_file']=str(out/f'strict_cost{bps}.log')
            for side in ['commission_buy','commission_sell']:
                stress['backtest']['transaction_cost'][side]+=bps/10000
            config=out/f'cost{bps}.yaml';config.write_text(yaml.safe_dump(stress,allow_unicode=True),encoding='utf-8')
            with (out/f'cost{bps}_console.log').open('w',encoding='utf-8') as log:
                cost_status[bps]=subprocess.run([sys.executable,str(Path(__file__).parent/'strict_run.py'),'--config',str(config)],stdout=log,stderr=subprocess.STDOUT).returncode
        for bps,exit_code in cost_status.items():
            if exit_code!=0:continue
            folder=out/('strict_backtest' if bps==0 else f'strict_cost{bps}')
            for factor in folder.iterdir():
                files=sorted(factor.rglob('period_return.csv'))
                if not files:continue
                r=pd.read_csv(files[-1],index_col=0)['Group10']
                calendar_table=pd.read_parquet(root/'data/source_data/Calendar.parquet')
                sessions=pd.DatetimeIndex(pd.to_datetime(calendar_table.loc[(calendar_table.Markettype==1)&(calendar_table.State=='O'),'Clddt'])).sort_values()
                expected_exits=[]
                for w in windows:
                    boundary=pd.Timestamp(w.test_periods[0])+pd.offsets.MonthEnd(1)
                    pos=sessions.searchsorted(boundary,side='right')
                    if pos>=len(sessions):raise ValueError('Unobserved terminal execution session')
                    expected_exits.append(sessions[pos])
                if not pd.DatetimeIndex(pd.to_datetime(r.index)).equals(pd.DatetimeIndex(expected_exits)) or not np.isfinite(r).all() or (r<=-1).any():raise ValueError('Invalid strict monthly return coverage')
                wealth=np.r_[1,np.cumprod(1+r)];vol=r.std(ddof=1)*np.sqrt(12);cagr=wealth[-1]**(12/len(r))-1
                trimmed=r.drop(r.nlargest(min(3,len(r)-2)).index)
                trading.append(dict(model=factor.name,extra_one_side_bps=bps,months=len(r),cagr=cagr,
                    sharpe=standard_sharpe(r,12,float(bt['backtest']['risk_free_rate'])),
                    sharpe_convention=CONVENTION,
                    max_drawdown=float((1-wealth/np.maximum.accumulate(wealth)).max()),
                    worst_month=float(r.min()),best_month=float(r.max()),
                    positive_month_fraction=float((r>0).mean()),
                    diagnostic_excluding_best3_mean=float(trimmed.mean()),
                    note='Excluding best months is a sensitivity diagnostic, not an alternative strategy return. No high-Sharpe threshold proves overfitting.'))
    if code==0 and len(trading)!=6:raise ValueError('Missing model or cost-scenario outputs')
    pd.DataFrame(trading).to_csv(out/'report/trading_summary.csv',index=False,encoding='utf-8-sig')
    save(out/'report/cost_status.json',cost_status)
    if any(sha(Path(f))!=h for f,h in hashes.items()):raise ValueError('Inputs changed during backtests')
    save(out/'report/status.json',dict(state='SMOKE_ONLY' if args.smoke else (('COMPLETE_PIT_REVIEW_REQUIRED' if timing_review['review_required'] else 'COMPLETE') if code==0 and all(v==0 for v in cost_status.values()) else 'PREDICTIVE_COMPLETE_TRADING_BLOCKED'),
        strict_exit_code=code,feature_timing_review_required=timing_review['review_required'],production_modified=False,output=str(out),note='Never use smoke metrics as evidence. Trading failure is not zero return.'))
    print('Completed. Read report/status.json before interpreting any performance.',flush=True)


if __name__=='__main__':main()
