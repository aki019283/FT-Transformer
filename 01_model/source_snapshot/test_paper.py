import unittest
import numpy as np
import pandas as pd
import torch
from paper_model import PaperFT,optimizer_groups,safe_features,audit_raw_features,rank64_frame
import tempfile
from pathlib import Path
import json

class Tests(unittest.TestCase):
 def test_rank64_large_values_ties_missing_and_month_isolation(self):
  cfg={'dataset':[{'kind':'sanitize'},{'kind':'cross_section_rank','params':{'output_range':[-1.0,1.0],'rank_target':True}}]}
  frame=pd.DataFrame({'TradingDate':['a']*5+['b']*2,'x':[1e39,1e61,2.,2.,np.nan,1e61,1e39],'y':[1.,2.,3.,4.,np.nan,1.,2.]})
  result=rank64_frame(frame,['x'],'y','TradingDate',cfg)
  np.testing.assert_allclose(result.x,[.5,1.,-.25,-.25,np.nan,1.,0.],equal_nan=True)
  self.assertEqual(result.x.dtype,np.dtype('float32'))
  with tempfile.TemporaryDirectory() as d:
   audit_raw_features(frame,['x'],Path(d)/'audit.json',allow_rank64=True)
   frame.loc[0,'x']=np.inf
   with self.assertRaises(ValueError):audit_raw_features(frame,['x'],Path(d)/'audit.json',allow_rank64=True)
  with self.assertRaises(ValueError):rank64_frame(frame,['x'],'y','TradingDate',cfg)
 def test_raw_audit_before_sanitize(self):
  with tempfile.TemporaryDirectory() as d:
   dest=Path(d)/'audit.json'
   frame=pd.DataFrame({'TradingDate':['2020-01-01']*3,'Stkcd':['1','2','3'],'factor':[1e100,np.inf,np.nan]})
   original=frame.copy(deep=True)
   with self.assertRaises(ValueError):audit_raw_features(frame,['factor'],dest)
   report=json.loads(dest.read_text())
   self.assertEqual(report['columns'][0]['finite_float32_overflow'],1)
   self.assertEqual(report['columns'][0]['infinite'],1)
   pd.testing.assert_frame_equal(frame,original)
   audit_raw_features(frame.iloc[2:],['factor'],dest)
 def test_structure_and_gradients(self):
  torch.set_num_threads(2);torch.manual_seed(42)
  model=PaperFT(100)
  self.assertEqual(len(model.core.layers),3)
  self.assertNotIn('norm0',model.core.layers[0])
  self.assertIn('norm0',model.core.layers[1])
  self.assertTrue(920000<sum(p.numel() for p in model.parameters())<940000)
  x=torch.randn(4,100);y=model(x)
  self.assertEqual(tuple(y.shape),(4,));y.square().mean().backward()
  self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
  model.eval();self.assertTrue(torch.equal(model(x),model(x)))
 def test_tokenizer(self):
  model=PaperFT(2);x=torch.tensor([[2.,3.]])
  t=model.core.tokenizer;actual=t(x,None)
  expected=torch.cat([torch.ones(1,1),x],dim=1).unsqueeze(-1)*t.weight
  expected[:,1:]+=t.bias
  torch.testing.assert_close(actual,expected)
 def test_decay(self):
  core=PaperFT(2).core;groups=optimizer_groups(core)
  ids=[id(p) for g in groups for p in g['params']]
  self.assertEqual(len(ids),len(set(ids)))
  self.assertEqual(set(ids),{id(p) for p in core.parameters()})
  self.assertIn(id(core.tokenizer.weight),{id(p) for p in groups[1]['params']})
 def test_feature_safety(self):
  for value in [np.inf,-np.inf,1e100]:
   with self.assertRaises(ValueError):safe_features(pd.DataFrame({'x':[value]}))
  np.testing.assert_array_equal(safe_features(pd.DataFrame({'x':[np.nan,1]})),[[0],[1]])

if __name__=='__main__':unittest.main()
