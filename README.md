# 层合板渐进失效计算程序

对任意材料、厚度和角度组成的层合板进行经典层合板理论计算，支持膜力、弯矩及其组合的比例加载。Hoffman 准则触发逐层损伤，退化后立即在相同载荷下重新求解，整体破坏以所有单层板完全失效为准。

## 快速开始

需要 Python 3.10 或更新版本。在本目录运行：

```powershell
python -m pip install -r requirements.txt
python main.py examples/tension.json --output results/tension
python main.py examples/bending.json --output results/bending
python main.py examples/cascade.json --output results/cascade
```

结果保存在指定目录：

| 文件 | 内容 |
| --- | --- |
| `report.txt` | 中文输入摘要、初始 ABD、首层失效、逐次损伤及最终状态 |
| `history.csv` | 每个可解状态的载荷、六分量应变/曲率、损伤状态及事件 |
| `load_strain.png` | 纵向中面应变 εx⁰ 与加载倍数 λ 的关系 |

未指定 `--output` 时写入 `results`。重复运行会覆盖该目录内同名结果文件。输入或计算无效时返回非零退出码并给出原因；“无有限后续失效根”属于有效分析结果，通过报告说明。

## 配置输入

复制示例 JSON 后修改材料、铺层和参考载荷。最小输入示例：

```json
{
  "title": "我的层合板",
  "materials": {
    "material_A": {
      "E1": 45000, "E2": 12000, "nu12": 0.25, "G12": 5000,
      "Xt": 1000, "Xc": 700, "Yt": 40, "Yc": 150, "S": 60
    }
  },
  "plies": [
    {"material": "material_A", "thickness": 0.125, "angle": 0},
    {"material": "material_A", "thickness": 0.125, "angle": 90}
  ],
  "reference_load": [1, 0, 0, 0, 0, 0]
}
```

- `title` 可省略；其余顶层字段必填。材料字段使用 `nu12` 表示 ν12。材料名称由使用者指定，每一层可引用不同材料。
- 模量、强度单位为 **MPa**，厚度单位为 **mm**；所有拉伸、压缩和剪切强度均填正数。
- `plies` 按**上表面至下表面**顺序输入，第 1 层为最上层，层号向下递增。总厚度中面为 `z=0`，**向下为正**，最上表面 `z=-h/2`，最下表面 `z=+h/2`。角度以度为单位，从全局 x 轴逆时针转向材料 1 轴；支持负角度。
- `reference_load` 顺序为 **[Nx, Ny, Nxy, Mx, My, Mxy]**。前三项单位 **N/mm**，后三项为单位宽度弯矩，单位 **N**，不能把板的总外力直接填入。
- 实际载荷为 `λ × reference_load`，从 λ=0 单调增大。六项可以正负混合，但不能全为零；纯弯曲可以填 `[0, 0, 0, 1, 0, 0]`。
- 文档中的 `[N, αN, βN, aN, bN, cN]` 可通过适当的参考载荷直接表达；程序也支持 Nx=0。
- 使用工程剪应变 γxy 和扭曲率 κxy：`ε(z)=ε⁰+zκ`，应变无量纲，曲率单位为 **1/mm**。
- 弯矩合力定义为 `M=∫zσ dz`，正曲率表示对应应变沿向下的 z 方向增大。例如单层 0° 板施加正 Mx 纯弯曲时，上表面纵向受压，下表面纵向受拉。

旧版本采用 z 向上为正、从下向上输入。若要保持旧输入所代表的同一块物理层合板及同一载荷，应将 `plies` 顺序反转，并将 Mx、My、Mxy 全部取反；膜力分量和铺层角度保持原值。对应中面应变不变、曲率与 z 坐标取反、层号重新从顶部编号。本目录示例现在直接按新约定解释原有数组及载荷数值，不自动执行旧数据转换。

示例材料仅用于演示，不代表某种实际材料的试验数据。


## 查看结果与手算复核

曲线以 εx⁰ 为横轴、λ 为纵轴。同一 λ 的相邻点表示退化和载荷重新分配造成的应变跳变；不同载荷之间在线性损伤状态内连线。最后的整体失效或计算中止标记对应最后一次退化前的可解状态。允许载荷可平衡的奇异刚度状态继续计算，输出的应变是最小范数代表解，零刚度方向的变形不唯一。纯弯曲时 εx⁰ 可能为零，曲率请查看 CSV。

`history.csv` 使用 UTF-8 BOM，方便 Excel 打开。`states` 为本行求解时的各层状态；`events` 是该行触发的事件 JSON 数组，包含层号、位置、应力、利用率、Hoffman 指数及退化前后状态。`load_step` 表示增载事件序号；`cascade_step=0` 表示首次增载触发，正值表示在该载荷下的后续检查。`stiffness_rank` 为厚度缩放后的 ABD 数值秩，小于 6 表示采用最小范数代表解。每行均对应一个可解状态，所有层完全失效或载荷不再能平衡时不添加新的平衡点。

整体破坏要求**所有层均为 `fully_failed`，即所有层 Q 均为零**。横向失效后仅保留 Q11 或剪切失效均为部分损伤，不算该层完全破坏。退化后对厚度缩放的 ABD 做特征分解，在非零刚度子空间求解；取 `[ε⁰,hκ]` 在零刚度方向的分量为零。只有参考载荷在零刚度方向的相对分量不超过 `1e-10`，且平衡残差通过检查，才允许继续。秩判定采用最大特征值的 `1e-12` 为阈值。

结果状态为 `all_plies_failed` 时才设置 `ultimate_load_factor`。如尚有未完全失效层，但载荷已无法平衡，则返回 `load_unsupported`，`terminal_load_factor` 给出计算中止载荷，`ultimate_load_factor=None`；不把未失效层强制清零。数值求解失败返回 `numerical_failure`，无下一有效失效根返回 `no_finite_failure`，均不认定所有层破坏。初始刚度仍须正定有效。

`cascade.json` 是可手算验证：四层厚度均为 1 mm、纵向模量均为 10000 MPa、泊松比为 0；外侧两层 Xt=10 MPa，中间两层 Xt=15 MPa。

- 当 Nx=40 N/mm 时，各层应力均为 10 MPa，外侧两层同时失效，εx⁰=0.001。
- 保持 Nx=40 N/mm，剩余两层应力增至 20 MPa，εx⁰=0.002，超过 15 MPa 后连续失效。
- 首层与整体失效加载倍数均为 40，记录顺序为第 1、4 层，然后第 2、3 层。

## Python 调用与测试

```python
from laminate import Laminate, analyze
from reporting import write_outputs

model = Laminate.from_json("examples/tension.json")
result = analyze(model)
print(result.first_failure_load_factor, result.ultimate_load_factor)
for event in result.events:
    print(event.load_factor, event.ply, event.mode, event.cascade_step)
write_outputs(result, "results/custom")
```

也可用 `Laminate.from_dict(data)` 或 `Material`、`Ply`、`Laminate` 数据类构造模型。`analyze` 不写文件也不修改输入；返回 `AnalysisResult`，包含事件、历程、最终损伤状态、初始 ABD 和终止原因。层号从 1 开始；Python 结果中的 `history_index` 和 `event_indices` 从 0 开始。

`interfaces` 按从上到下顺序给出界面 z 坐标；`surface_stresses` 的第二维顺序为 `[上表面, 下表面]`。事件和 CSV 中的 `surface="top"` 表示本层上表面，`"bottom"` 表示本层下表面。同层两面均失效时取更严重的模式，同模式取 Hoffman 较大的一面，完全并列取上表面。

```powershell
python -m unittest discover -s tests -v
```

测试包括手算刚度、角度变换、拉弯耦合、拉压与剪切强度、模式选择、退化、同时/连续失效、参考载荷缩放、异常输入和全部输出文件。

## 玻璃/环氧课堂算例检验

`examples/glass_cross_ply.json` 对应 [0/90/0] 对称板，0°层总厚度与 90°层总厚度比为 0.2。总厚度归一化为 1 mm，上至下各层厚度为 1/12、5/6、1/12 mm。

运行 `python verify_cross_ply.py` 可生成标准输出、与题图参考值的比较图，以及中文核验说明，保存在 `results/glass_cross_ply`。脚本用独立的二维膜力平衡解析式复核应变、应力和失效载荷。

当前模型在 `Nx/t=38.361502753 MPa` 时，中间 90°层横向失效，两侧 0°层在同一载荷下连续横向失效。剪切刚度消失后继续计算，到 `Nx/t=172.333333333 MPa` 时两侧 0°层纵向完全失效。90°层仍有全局 y 方向刚度，却无法承受 Nx，因此以 `load_unsupported` 中止，尚未达到所有层完全失效，不给出该准则的整体破坏载荷。

## 计算原理
见 [原理.md](note/原理.md) 
