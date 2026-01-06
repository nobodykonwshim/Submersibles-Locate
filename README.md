# 潜航器 Locate 模块

本仓库实现 Locate（Leader）模块：读取预处理观测数据，融合位置并估计不确定性，输出面向后续团队的标准接口文件。

## 环境要求
- Python 3.10+
- 依赖：`pandas`、`numpy`、`openpyxl`、`matplotlib`（用于可视化）、`pillow`（用于 GIF 动画导出）

安装依赖（如需）：
```bash
pip install -r requirements.txt  # 或手动安装上述包
```

## 配置
根目录的 `config.json` 包含全部可调参数、随机种子以及接口设置。主要字段：
- `seed`：随机种子，用于样例数据可复现。
- `chi2_2d_95`：二维 95% 置信的卡方值（默认 5.9914645471）。
- `r95_min_km`：R95 半径下限。
- `bbox_min_km`、`bbox_multiplier_on_r95`：控制搜索区域边界框尺寸。
- `depth_bins_m`：深度分箱边界（米）。
- `smoothing_alpha`、`residual_ewma_alpha`：位置与残差协方差的 EWMA 系数。
- `depth_prior_default`：在没有深度样本时使用的先验深度分布。

## 运行方式
在仓库根目录执行：

1. 生成可复现的样例观测（可选）：
```bash
python main.py --generate-sample-data
```
会创建 `data/processed_v1/observations.csv`。

2. 运行 locate 阶段并生成输出：
```bash
python main.py --stage locate
```
输出写入 `outputs/locate/`：
- `track.csv`
- `summary.json`
- `Locate_to_Member2_Interface.xlsx`（含 4 个工作表）
- `R95_TimeSeries.csv`、`SearchRegion.csv`、`DepthDistribution.csv`、`EquipmentTemplate.csv`
- `visualizations/r95_timeseries.png`：R95 随时间变化曲线
- `visualizations/track_animation.gif`：平滑位置随时间演化的动画，用于验证动态轨迹

在无外部数据时，顺序运行上述两条命令即可使用默认配置生成可复现结果。
