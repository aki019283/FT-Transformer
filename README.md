# FT-Transformer 多因子选股实验

本目录保存 seed 42 实验的训练和回测源码。对应版本在 2024 年 2 月至 2025 年 11 月的 22 个持有月内，以固定持仓、月初首个交易日收盘至下月首个交易日收盘重估，沿用简化交易费用，月度净收益的标准年化夏普为 1.476659537079。该数值只对应上述历史评价区间与执行假设。

## 入口

- 训练和预处理入口为 `01_model/source_snapshot/study.py`，模型适配层为同目录的 `paper_model.py`。原训练协议与参数保存在 `protocol.json`。运行完整历史训练器时会执行其中登记的基线与种子，本次提交对应的评分为 FT seed 42。
- 原项目框架及 YAML 配置位于 `01_model/project`。训练入口需要项目的行情、股票池和因子面板，目录位置按配置加载。
- 对应 1.477 的回测入口为 `02_backtest/replay.py`。使用服务器交付包中的 `03_predictions` 与 `04_results`，放在本文件同级，随后在仓库此目录执行下列命令。请将数据路径换成实际行情目录。输出路径由脚本新建。
- 论文作者的实现、许可和文件校验记录分别位于 `01_model/source_snapshot/vendor` 和 `upstream.json`。

```powershell
$out = Join-Path $env:TEMP ('ft1477_' + [guid]::NewGuid().ToString('N'))
python -B -X utf8 .\02_backtest\replay.py --data 'F:\aki\mlp_factor\data\source_data' --output $out
```
