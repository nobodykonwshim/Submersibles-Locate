# 潜航器 Locate 模块

本仓库实现 Locate（Leader）模块：读取预处理观测数据，融合 3D 位置并估计不确定性，输出面向后续团队的标准接口文件，并生成带海洋环境的可视化成果。

## 环境要求
- Python 3.10+
- 依赖：`pandas`、`numpy`、`openpyxl`、`matplotlib`（用于可视化）、`pillow`（用于 GIF 动画导出）、`scipy`

安装依赖（如需）：
```bash
pip install -r requirements.txt  # 或手动安装上述包
```

## 配置
根目录的 `config.json` 包含全部可调参数、随机种子以及接口设置。主要字段：
- `seed`：随机种子，用于样例数据和海洋场的可复现。
- `chi2_2d_95`、`chi2_3d_95`：95% 置信的卡方值（2D/3D）。
- `r95_min_km`：R95 半径下限。
- `bbox_min_km`、`bbox_multiplier_on_r95`：控制搜索区域边界框尺寸。
- `depth_bins_m`：深度分箱边界（米）。
- `smoothing_alpha`、`residual_ewma_alpha`：位置与残差协方差的 EWMA 系数。
- `depth_prior_default`：在没有深度样本时使用的先验深度分布。
- `ocean_grid_size`、`ocean_domain_km`：随机海洋网格大小与水平覆盖范围。
- `ocean_density_base`、`ocean_density_variation`：海水密度的基值与扰动幅度（驱动稳定性衰减）。
- `current_max_kmh`：用于合成表层流场的最大流速尺度。
- `bathymetry_max_depth_m`：随机海底地形的最大水深。

## 运行方式
工作目录需在仓库根目录（包含 `main.py`）。推荐先生成样例数据再运行 locate 阶段：

1. 生成可复现的样例观测（可选）：
```bash
python main.py --generate-sample-data
```
会创建 `data/processed_v1/observations.csv`，包含时间、二维观测、质量分数 `q` 与深度。

2. 运行 locate 阶段并生成输出：
```bash
python main.py --stage locate
```
输出写入 `outputs/locate/`：
- `track.csv`：3D 平滑轨迹及协方差、R95。
- `summary.json`：概要统计与搜索区域尺寸。
- `ocean_state.npz`：合成的海洋密度、流场和地形网格。
- `Locate_to_Member2_Interface.xlsx`（含 4 个工作表）及对应的 `R95_TimeSeries.csv`、`SearchRegion.csv`、`DepthDistribution.csv`、`EquipmentTemplate.csv`。
- 可视化：
  - `visualizations/r95_timeseries.png`：R95 随时间变化曲线。
  - `visualizations/track_static_3d.png`：3D 轨迹静态图（深度为负方向）。
  - `visualizations/track_animation_3d.gif`：3D 动态轨迹动画。
  - `visualizations/ocean_density_currents.png`：海水密度与表层流场。
  - `visualizations/ocean_bathymetry.png`：随机海底地形热力图。

在无外部数据时，顺序运行上述两条命令即可使用默认配置生成 3D 海洋环境下的可复现结果。
