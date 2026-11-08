"""Text, CSV and headless plot output for progressive laminate analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from laminate import AnalysisResult, FAILURE_TOL, STIFFNESS_RTOL


MODE_NAMES = {"fiber": "纵向失效", "transverse": "横向失效", "shear": "剪切失效"}
STATE_NAMES = {"intact": "完好", "shear_failed": "剪切失效", "transverse_failed": "仅保留 Q11",
               "fully_failed": "完全失效"}


def _vector(values) -> str:
    return "[" + ", ".join(f"{v:.10g}" for v in values) + "]"


def _loads(result: AnalysisResult, factor: float) -> np.ndarray:
    return np.asarray(result.model.reference_load) * factor


def write_report(result: AnalysisResult, path: str | Path) -> None:
    model = result.model
    lines = [model.title, "=" * 64, "输入与计算约定", "",
             "单位：材料模量和强度 MPa；厚度 mm；膜力 N/mm；弯矩合力 N。",
             "载荷顺序：[Nx, Ny, Nxy, Mx, My, Mxy]，实际载荷 = λ × 参考载荷。",
             f"参考载荷：{_vector(model.reference_load)}",
             f"铺层数：{len(model.plies)}；总厚度：{model.thickness:.10g} mm。",
             "铺层从上至下编号，第 1 层为最上层；中面 z=0，z 轴向下为正。",
             "上表面 z=-h/2，下表面 z=+h/2；角度从全局 x 轴逆时针转向材料 1 轴。",
             "工程剪应变约定，ε(z)=ε⁰+zκ。每层上下表面分别校核，界面两侧不合并。",
             "弯矩合力 M=∫zσ dz，正曲率表示对应应变沿向下的 z 方向增大。",
             "严格采用原文柔度交叉项 -nu12/E2（不是通常的 -nu12/E1）。",
             "Hoffman 达到 1 时判失效；单项超限优先级为纵向、横向、剪切。",
             "组合失效时按最大单项应力/强度比选择模式：这是补充的工程假设。",
             "同层两面均失效时采用更严重的模式；同模式取 Hoffman 指数较大的一面，完全并列取上表面。",
             "退化均针对材料坐标系 Q；横向失效后保留原始 Q11，纵向拉伸和压缩均保留。",
             "整体失效定义：所有单层板均完全失效，各层材料坐标系 Q 均为零。",
             "横向或剪切损伤不等于单层板完全失效；ABD 奇异本身不作为整体失效判据。",
             "退化后在剩余刚度子空间求解；若载荷无法平衡则单独终止，不认定所有层破坏。",
             "秩小于 6 时，输出取 [ε⁰,hκ] 的最小范数解；零刚度方向变形不唯一，取零作为代表。",
             f"失效判定容差：{FAILURE_TOL:g}；刚度相对特征值阈值：{STIFFNESS_RTOL:g}。",
             "未包含热湿载荷、几何非线性、屈曲、层间脱层及损伤断裂能。", "", "材料库"]
    for name, material in model.materials.items():
        values = ", ".join(f"{field}={getattr(material, field):.10g}" for field in material.__dataclass_fields__)
        lines.append(f"  {name}: {values}")
    lines.extend(["", "铺层（由上至下）"])
    interfaces = model.interfaces
    for i, ply in enumerate(model.plies):
        lines.append(f"  第 {i+1} 层：材料={ply.material}，厚度={ply.thickness:.10g} mm，"
                     f"角度={ply.angle:.10g}°，z=[{interfaces[i]:.10g}, {interfaces[i+1]:.10g}] mm")
    lines.extend(["", "初始 ABD（A: N/mm；B: N；D: N·mm）",
                  np.array2string(result.initial_abd, precision=9, suppress_small=False, max_line_width=120),
                  "", "计算摘要"])
    if result.first_failure_load_factor is None:
        lines.append("  首层失效：未得到有限载荷。")
    else:
        factor = result.first_failure_load_factor
        lines.append(f"  首层失效：λ={factor:.12g}，实际载荷={_vector(_loads(result, factor))}")
    if result.ultimate_load_factor is None:
        lines.append("  整体失效：未达到所有单层板完全失效，未确定该准则的整体失效载荷。")
    else:
        factor = result.ultimate_load_factor
        lines.append(f"  整体失效：λ={factor:.12g}，实际载荷={_vector(_loads(result, factor))}")
    lines.extend([f"  状态：{result.status}", f"  终止原因：{result.reason}",
                  f"  层损伤记录数：{len(result.events)}", "", "按加载顺序的破坏记录"])
    if result.terminal_load_factor is not None and result.ultimate_load_factor is None:
        lines.insert(-2, f"  计算中止：λ={result.terminal_load_factor:.12g}，"
                     f"实际载荷={_vector(_loads(result, result.terminal_load_factor))}（不是全层破坏载荷）")
    for i, event in enumerate(result.events, 1):
        continuous = "是（同载荷重新分配后）" if event.cascade_step > 0 else "否（增载触发）"
        surface = "下表面" if event.surface == "bottom" else "上表面"
        direction = ""
        if event.mode in ("fiber", "transverse"):
            stress = event.stress[0 if event.mode == "fiber" else 1]
            direction = "（拉伸）" if stress >= 0 else "（压缩）"
        lines.extend([
            "", f"事件 {i} | 加载步骤 {event.load_step} | 连续失效子步骤 {event.cascade_step}",
            f"  λ={event.load_factor:.12g}；实际载荷={_vector(_loads(result, event.load_factor))}",
            f"  第 {event.ply} 层，{surface}，z={event.z:.10g} mm；{MODE_NAMES[event.mode]}{direction}",
            f"  [σ1, σ2, τ12]={_vector(event.stress)} MPa；Hoffman={event.failure_index:.12g}",
            f"  [纵向, 横向, 剪切] 利用率={_vector(event.utilization)}",
            f"  {STATE_NAMES[event.state_before]} → {STATE_NAMES[event.state_after]}；连续失效：{continuous}",
            f"  退化前 [εx⁰, εy⁰, γxy⁰, κx, κy, κxy]={_vector(result.history[event.history_index].response)}",
        ])
    lines.extend(["", "最终损伤状态"])
    lines.extend(f"  第 {i} 层：{STATE_NAMES[state]}" for i, state in enumerate(result.final_states, 1))
    lines.extend(["", "曲线说明：横轴 εx⁰，纵轴 λ。相同 λ 的相邻点表示退化后的应变跳变。",
                  "达到整体失效时只保留退化前可解的状态，不构造失效后的平衡点。",
                  "history.csv 的每行是一次可解状态；events 列为该状态触发的损伤事件 JSON 数组。",
                  "stiffness_rank 为厚度缩放后 ABD 的数值秩；小于 6 的行使用最小范数平衡代表解。",
                  "κ 的单位为 1/mm；states 列记录本行计算时（本批退化前）的各层损伤状态。", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_history(result: AnalysisResult, path: str | Path) -> None:
    columns = ["load_factor", "Nx", "Ny", "Nxy", "Mx", "My", "Mxy",
               "epsilon_x0", "epsilon_y0", "gamma_xy0", "kappa_x", "kappa_y", "kappa_xy",
               "stage", "load_step", "cascade_step", "states", "events", "stiffness_rank"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for point in result.history:
            event_data = []
            for index in point.event_indices:
                event = result.events[index]
                event_data.append({"event": index+1, "ply": event.ply, "surface": event.surface,
                                   "z": event.z, "mode": event.mode, "stress": event.stress,
                                   "hoffman": event.failure_index, "utilization": event.utilization,
                                   "state_before": event.state_before, "state_after": event.state_after})
            writer.writerow([point.load_factor, *_loads(result, point.load_factor), *point.response,
                             point.stage, point.load_step, point.cascade_step,
                             json.dumps(point.states, ensure_ascii=False),
                             json.dumps(event_data, ensure_ascii=False, allow_nan=False), point.stiffness_rank])


def plot_curve(result: AnalysisResult, path: str | Path) -> None:
    # Use a standalone canvas: no GUI and no process-wide backend change for API callers.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.2, 4.8), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    strain = np.array([point.response[0] for point in result.history])
    factors = np.array([point.load_factor for point in result.history])
    ax.plot(strain, factors, color="#27364b", linewidth=1.6, marker=".", markersize=4)
    if result.events:
        first_index = result.events[0].history_index
        ax.scatter(strain[first_index], factors[first_index], s=65, facecolors="white",
                   edgecolors="#2375ad", linewidths=1.7, zorder=4,
                   label=f"First ply failure: {result.first_failure_load_factor:.6g}")
    if result.ultimate_load_factor is not None:
        ax.scatter(strain[-1], factors[-1], s=65, marker="x", color="#c0392b", linewidths=1.7,
                   zorder=5, label=f"All plies fully failed: {result.ultimate_load_factor:.6g}")
    elif result.terminal_load_factor is not None:
        stop_label = "Load no longer supported" if result.status == "load_unsupported" else "Numerical solution stopped"
        ax.scatter(strain[-1], factors[-1], s=65, marker="x", color="#c0392b", zorder=5,
                   label=f"{stop_label}: {result.terminal_load_factor:.6g}")
    else:
        ax.text(.03, .96, "No finite subsequent failure load\nOnly computed states are shown",
                transform=ax.transAxes, va="top", fontsize=9, color="#555555")
    if result.events:
        ax.legend(loc="best", frameon=False, fontsize=9)
    ax.set_xlabel(r"Mid-plane longitudinal strain $\varepsilon_x^0$")
    ax.set_ylabel(r"Load multiplier $\lambda$")
    ax.set_title("Laminate progressive failure", fontsize=12)
    if any(point.stiffness_rank < 6 for point in result.history):
        fig.text(.5, .005, "Singular states use minimum-norm equilibrium; free-mode deformation is undetermined.",
                 ha="center", fontsize=7)
        fig.get_layout_engine().set(rect=(0, .04, 1, .96))
    ax.set_ylim(bottom=0)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useMathText=True)
    ax.grid(True, color="#e7e9ed", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=180, facecolor="white")
    fig.clear()


def write_outputs(result: AnalysisResult, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_report(result, directory / "report.txt")
    write_history(result, directory / "history.csv")
    plot_curve(result, directory / "load_strain.png")
