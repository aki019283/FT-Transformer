import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'strict_src'))
from backtest.config import LoadBacktestConfig
from backtest.runner import BacktestRunner
import argparse
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
BacktestRunner(LoadBacktestConfig(a.config)).Run()
