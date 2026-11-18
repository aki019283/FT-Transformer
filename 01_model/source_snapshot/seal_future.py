"""Freeze six final checkpoints; seal real forecasts before first monthly session.
No training, fabricated predictions or future return reads occur here.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
 p.add_argument('--predictions',type=Path);p.add_argument('--month');p.add_argument('--calendar',type=Path)
 a=p.parse_args();now=pd.Timestamp.now(tz='Asia/Shanghai')
 registration=a.run/'preregistration.json'
 registered=json.loads(registration.read_text(encoding='utf8'))
 if registered['smoke']:raise ValueError('Cannot freeze smoke models')
 folder=a.run/'future_sealed';folder.mkdir(exist_ok=True)
 freeze=folder/'frozen_models.json'
 checkpoints={}
 for m in ['mlp','ft_transformer']:
  for s in [42,123,2026]:
   windows=sorted((a.run/'windows'/f'{m}_s{s}').glob('*/COMMITTED.json'))
   if len(windows)!=68:raise ValueError('Require all 68 historical windows per model/seed')
   marker=windows[-1];manifest=json.loads(marker.read_text())
   for name,h in manifest.items():
    if sha(marker.parent/name)!=h:raise ValueError('Checkpoint mismatch')
   checkpoint=marker.parent/'model.pt';checkpoints[str(checkpoint.resolve())]=sha(checkpoint)
 payload=dict(checkpoints=checkpoints,registration_sha256=sha(registration))
 if not freeze.exists():
  if now>=pd.Timestamp('2027-01-01',tz='Asia/Shanghai'):raise ValueError('Missed prospective model-freeze deadline')
  with freeze.open('x',encoding='utf8') as f:json.dump(dict(payload,sealed_at=str(now)),f,indent=2)
 else:
  previous=json.loads(freeze.read_text());previous.pop('sealed_at')
  if previous!=payload:raise ValueError('Frozen model changed')
 if a.predictions:
  if not a.month or not a.calendar:raise ValueError('Need --month YYYY-MM and --calendar')
  month=pd.Period(a.month,'M')
  if month.year!=2027:raise ValueError('Only registered future year allowed')
  cal=pd.read_parquet(a.calendar)
  dates=pd.to_datetime(cal.loc[(cal.Markettype==1)&(cal.State=='O'),'Clddt'])
  dates=dates[dates.dt.to_period('M')==month]
  if dates.empty:raise ValueError('Missing future trading calendar')
  deadline=pd.Timestamp(dates.min()).normalize().tz_localize('Asia/Shanghai')+pd.Timedelta(hours=9,minutes=30)
  if now>=deadline:raise ValueError('Late forecast; cannot label prospective')
  df=pd.read_parquet(a.predictions)
  columns=['TradingDate','Stkcd','mlp_ensemble','ft_transformer_ensemble']
  if list(df.columns)!=columns or df.empty or df.duplicated(columns[:2]).any():raise ValueError('Invalid prediction schema/keys')
  import numpy as np
  if not np.isfinite(df[columns[2:]].to_numpy()).all():raise ValueError('Nonfinite forecasts')
  if not (pd.to_datetime(df.TradingDate).dt.to_period('M')==month).all():raise ValueError('Wrong holding month')
  dest=folder/(str(month)+'.parquet')
  with dest.open('xb') as f:f.write(a.predictions.read_bytes())
  with (folder/(str(month)+'.json')).open('x',encoding='utf8') as f:
   json.dump(dict(sealed_at=str(now),deadline=str(deadline),sha256=sha(dest),freeze_sha256=sha(freeze),
    qualification='Timestamp and file checks only; feature availability and use of frozen checkpoints require source audit'),f,indent=2)
 print('Future seal verified:',folder)
if __name__=='__main__':main()
