"""Reproduce the supplied classroom example and independently check membrane equilibrium.

Run: python verify_cross_ply.py
The independent calculation uses explicit 2x2 formulas, without the production
stiffness assembly, coordinate transformation, root solver or response solver.
"""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from laminate import Laminate, analyze
from reporting import write_outputs


ROOT = Path(__file__).resolve().parent


def independent_solution(model, states, factor, standard=False):
    m = model.materials["glass_epoxy"]
    # Direct inverse of the 2x2 normal compliance block.
    s11, s22 = 1/m.E1, 1/m.E2
    s12 = -m.nu12/(m.E1 if standard else m.E2)
    determinant = s11*s22-s12*s12
    original = np.array([[s22, -s12], [-s12, s11]])/determinant
    local_q = []
    a = np.zeros((2, 2))
    for ply, state in zip(model.plies, states):
        q = original.copy()
        if state == "transverse_failed":
            q = np.diag([original[0, 0], 0.])
        elif state == "fully_failed":
            q[:] = 0
        elif state not in ("intact", "shear_failed"):
            raise ValueError(state)
        local_q.append(q)
        global_q = q if ply.angle == 0 else q[::-1, ::-1]
        a += ply.thickness*global_q
    denominator = a[0, 0]*a[1, 1]-a[0, 1]**2
    strain = factor*np.array([a[1, 1], -a[0, 1]])/denominator
    stress = np.array([q @ (strain if ply.angle == 0 else strain[::-1])
                       for ply, q in zip(model.plies, local_q)])
    return a, strain, stress


def independent_fi(stress, m):
    s1, s2 = stress
    return ((s1*s1-s1*s2)/(m.Xt*m.Xc) + s2*s2/(m.Yt*m.Yc)
            + (m.Xc-m.Xt)*s1/(m.Xt*m.Xc) + (m.Yc-m.Yt)*s2/(m.Yt*m.Yc))


def independent_first_failure(model, standard=False):
    _, _, stress = independent_solution(model, ("intact",)*3, 1., standard)
    m = model.materials["glass_epoxy"]
    candidates = []
    for s1, s2 in stress:
        a = (s1*s1-s1*s2)/(m.Xt*m.Xc)+s2*s2/(m.Yt*m.Yc)
        b = (m.Xc-m.Xt)*s1/(m.Xt*m.Xc)+(m.Yc-m.Yt)*s2/(m.Yt*m.Yc)
        candidates.extend(float(r.real) for r in np.roots([a, b, -1])
                          if abs(r.imag) < 1e-12 and r.real > 0)
    return min(candidates)


def verify(model, result):
    independent_load = independent_first_failure(model)
    np.testing.assert_allclose(result.first_failure_load_factor, independent_load, rtol=1e-11)
    assert result.ultimate_load_factor is None
    assert result.status == "load_unsupported"
    np.testing.assert_allclose(result.terminal_load_factor, model.materials["glass_epoxy"].Xt/6, rtol=1e-11)
    assert [(e.ply, e.mode, e.cascade_step) for e in result.events] == [
        (2, "transverse", 0), (1, "transverse", 1), (3, "transverse", 1),
        (1, "fiber", 0), (3, "fiber", 0)]
    for point in result.history:
        _, strain, stresses = independent_solution(model, point.states, point.load_factor)
        np.testing.assert_allclose(point.response[:2], strain, rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(point.response[2:], 0., atol=1e-12)
        for event_index in point.event_indices:
            event = result.events[event_index]
            np.testing.assert_allclose(event.stress[:2], stresses[event.ply-1], rtol=1e-11)
            np.testing.assert_allclose(event.failure_index,
                                       independent_fi(stresses[event.ply-1], model.materials["glass_epoxy"]),
                                       rtol=1e-11)


def main():
    model = Laminate.from_json(ROOT / "examples/glass_cross_ply.json")
    result = analyze(model)
    verify(model, result)
    output = ROOT / "results/glass_cross_ply"
    write_outputs(result, output)
    standard_first = independent_first_failure(model, standard=True)
    _, redistributed_strain, redistributed_stress = independent_solution(
        model, ("intact", "transverse_failed", "intact"), result.first_failure_load_factor)
    percentage_error = (result.first_failure_load_factor/36.677-1)*100
    # Keep the photographed labels literal; do not silently repair 0.0153 to 0.153.
    reference_x = [0, .0153, .655, 1.877]
    reference_y = [0, 36.677, 82.552, 194.485]
    fig = Figure(figsize=(8, 5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot([p.response[0]*100 for p in result.history],
            [p.load_factor/model.thickness for p in result.history],
            "o-", color="#176a94", label="Current software (Hoffman, surviving stiffness)")
    ax.plot(reference_x, reference_y, "x--", color="#a75938", label="Figure 2: transcribed labels")
    ax.annotate("Axial load unsupported;\n90-degree ply not fully failed",
                xy=(result.history[-1].response[0]*100, result.terminal_load_factor),
                xytext=(.65, 125), arrowprops={"arrowstyle": "->", "color": "#176a94"}, fontsize=9)
    ax.text(.03, .96, "Reference first strain is printed as 0.0153%;\npossible decimal typo (not corrected here).",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlabel(r"Mid-plane longitudinal strain $\varepsilon_x^0$ (%)")
    ax.set_ylabel(r"Average axial stress $N_x/t$ (MPa)")
    ax.set_title("Glass/epoxy cross-ply: [0/90/0], t0/t90 = 0.2")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.2)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.clear()
    lines = [
        "# 图一玻璃/环氧正交层合板检验", "",
        "## 输入与归一化", "",
        "用户确认 M=t0/t90=0.2，其中 t0 为全部 0°层总厚度。按图示对称 [0/90/0]，两侧 0°层等厚。",
        "取总厚度 t=1 mm，则由上至下厚度为 1/12、5/6、1/12 mm。z 轴向下为正，中面 z=0。",
        "图一 GPa 数值转换为 MPa：E1=53740、E2=17950、G12=8626、ν12=0.25；",
        "Xt=Xc=1034、Yt=27、Yc=138、S=41。仅施加 Nx，参考载荷为 [1,0,0,0,0,0]。",
        "因为 t=1 mm，加载倍数的数值等于 Nx/t（MPa）。任意其他总厚度下应使用 Nx=(Nx/t)×t（N/mm）。",
        "以下使用当前程序已确认的原文柔度交叉项 -ν12/E2。", "",
        "## 程序结果", "",
        "| 顺序 | Nx/t (MPa) | εx⁰ (%)，退化前 | 层号 | 模式 | σ1 / σ2 (MPa) | Hoffman |",
        "| --- | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for number, event in enumerate(result.events, 1):
        lines.append(f"| {number} "
                     f"| {event.load_factor:.9f} | {result.history[event.history_index].response[0]*100:.9f} "
                     f"| {event.ply} | {'纵向完全失效' if event.mode == 'fiber' else '横向失效'} "
                     f"| {event.stress[0]:.6f} / {event.stress[1]:.6f} "
                     f"| {event.failure_index:.9f} |")
    lines += ["", "第 2 层先横向失效，保留其材料 1 方向刚度（全局 y 方向）。",
              "在相同 Nx 下重新分配载荷后，第 1、3 层同时横向失效，仍保留全局 x 方向刚度。",
              "此时所有层的剪切刚度都为零，ABD 秩降为 4；当前 Nx 仍能平衡，在剩余刚度子空间继续加载。",
              "在 Nx/t=1034/6=172.333333333 MPa、εx⁰=1.564051043% 时，两侧 0°层纵向完全失效。",
              "第 2 层仍保留全局 y 方向刚度，但不能承受 Nx，返回 load_unsupported。",
              "**新整体失效准则要求所有层 Q 均为零；本算例未达到此条件，不输出全层破坏载荷。**",
              "秩亏状态使用厚度缩放后的最小范数平衡代表解，零刚度方向的变形不唯一。", "",
              "## 独立核算", "",
              "校验脚本独立使用 2×2 正应力柔度的解析逆矩阵、0°/90°分量交换及显式膜力平衡；",
              "不调用计算核心的刚度组装、坐标转换、响应求解或失效根函数。",
              "对每个可解状态的中面应变、事件局部应力和 Hoffman 指数逐项比较，相对误差容差 1e-11，均通过。",
              f"独立求得首层失效 Nx/t={independent_first_failure(model):.12g} MPa。",
              f"第 2 层退化后的两侧应力 σ1={redistributed_stress[0,0]:.9f}、σ2={redistributed_stress[0,1]:.9f} MPa；",
              "σ2 已超过 Yt=27 MPa，连续失效不是由加载步长造成的。", "",
              "## 与图二比较", "",
              "图二读数：N1/t=36.677 MPa、N2/t≈82.552 MPa、N3/t=194.485 MPa。",
              "中间载荷的印刷标点不清晰，这里按 82.552 读取。图二首个应变标签按可见字符读作 0.0153%，",
              "后两点为 0.655%、1.877%。首个值可能是 0.153% 的排印或识读问题，不能据此直接判软件错。",
              f"当前首层失效比 36.677 MPa 高 {percentage_error:.4f}%。后两点不具有相同终止定义，不能简单当作数值误差比较。",
              "", "差异来源：", "",
              f"1. **柔度公式**：独立对照仅将交叉项换成标准 -ν12/E1，首层值为 {standard_first:.9f} MPa，",
              "   更接近 36.677 MPa，但仍不相等。生产程序和算例材料参数未因此修改。",
              "2. **退化与停止条件**：当前程序以所有层完全失效为整体破坏准则，允许载荷可平衡的奇异刚度状态继续加载。",
              "   本例在两侧层完全失效后失去 Nx 承载能力，中止于 172.333333333 MPa；此时 90°层仍未完全失效。",
              "3. **失效判据及原始资料**：两张图不足以确认图二采用的失效判据、退化参数和材料数值精度，不能断言只改某一个公式即可复现。",
              "", "## 复现", "", "```powershell", "python verify_cross_ply.py",
              "python main.py examples/glass_cross_ply.json --output results/glass_cross_ply", "```", "",
              "比较图：comparison.png；完整计算报告：report.txt；逐状态数据：history.csv。", ""]
    (output / "verification.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Independent checks passed. First failure = {result.first_failure_load_factor:.12g} MPa")
    print(f"Standard-compliance diagnostic first failure = {standard_first:.12g} MPa")
    print(output)


if __name__ == "__main__":
    main()
