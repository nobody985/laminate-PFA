"""Classical laminate theory and proportional-load progressive failure.

Units: MPa, mm, N/mm (membrane forces), N (moment resultants).
Plies are numbered from top to bottom; z=0 at the mid-plane, positive downward.
Strain = mid-plane strain + z * curvature; moment resultants integrate z * stress.
The compliance deliberately uses -nu12/E2, as required by 原理.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


FAILURE_TOL = 1e-8
STIFFNESS_RTOL = 1e-12
EQUILIBRIUM_RTOL = 1e-10
MODES = ("fiber", "transverse", "shear")
STATES = ("intact", "shear_failed", "transverse_failed", "fully_failed")


def _number(value: object, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须是有限数值")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label} 必须是{'正的' if positive else ''}有限数值")
    return value


@dataclass(frozen=True)
class Material:
    E1: float
    E2: float
    nu12: float
    G12: float
    Xt: float
    Xc: float
    Yt: float
    Yc: float
    S: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _number(getattr(self, name), name, name != "nu12"))
        if not self.E2 / self.E1 > self.nu12**2:
            raise ValueError("原文柔度矩阵非正定：必须满足 E2 > nu12² × E1（交叉项为 -nu12/E2）")
        try:
            np.linalg.cholesky(self.compliance())
        except np.linalg.LinAlgError as exc:
            raise ValueError("原文柔度矩阵无法数值正定分解，请检查参数尺度") from exc

    def compliance(self) -> np.ndarray:
        return np.array([[1 / self.E1, -self.nu12 / self.E2, 0],
                         [-self.nu12 / self.E2, 1 / self.E2, 0],
                         [0, 0, 1 / self.G12]], dtype=float)

    def stiffness(self) -> np.ndarray:
        return np.linalg.solve(self.compliance(), np.eye(3))


@dataclass(frozen=True)
class Ply:
    material: str
    thickness: float
    angle: float

    def __post_init__(self) -> None:
        if not isinstance(self.material, str) or not self.material:
            raise ValueError("铺层 material 必须是非空材料名称")
        object.__setattr__(self, "thickness", _number(self.thickness, "thickness", True))
        object.__setattr__(self, "angle", _number(self.angle, "angle"))


@dataclass(frozen=True)
class Laminate:
    materials: Mapping[str, Material]
    plies: tuple[Ply, ...]
    reference_load: tuple[float, ...]
    title: str = "层合板渐进失效分析"

    def __post_init__(self) -> None:
        if not isinstance(self.title, str):
            raise ValueError("title 必须是字符串")
        if not self.materials or any(not isinstance(k, str) or not k or not isinstance(v, Material)
                                     for k, v in self.materials.items()):
            raise ValueError("materials 必须是非空的材料名称到 Material 的映射")
        if not self.plies or any(not isinstance(p, Ply) for p in self.plies):
            raise ValueError("plies 必须至少包含一层 Ply")
        for i, ply in enumerate(self.plies, 1):
            if ply.material not in self.materials:
                raise ValueError(f"第 {i} 层引用了不存在的材料：{ply.material}")
        if len(self.reference_load) != 6:
            raise ValueError("reference_load 必须包含 [Nx, Ny, Nxy, Mx, My, Mxy] 六个分量")
        load = tuple(_number(v, f"reference_load[{i}]") for i, v in enumerate(self.reference_load))
        if not any(load):
            raise ValueError("参考载荷不能全部为零")
        object.__setattr__(self, "reference_load", load)
        object.__setattr__(self, "plies", tuple(self.plies))
        object.__setattr__(self, "materials", dict(self.materials))
        if not math.isfinite(self.thickness):
            raise ValueError("总厚度超出数值范围")

    @property
    def thickness(self) -> float:
        return sum(p.thickness for p in self.plies)

    @property
    def interfaces(self) -> np.ndarray:
        """Top-to-bottom interface coordinates, from -h/2 to +h/2."""
        return np.concatenate(([0.0], np.cumsum([p.thickness for p in self.plies]))) - self.thickness / 2

    @classmethod
    def from_dict(cls, data: dict) -> Laminate:
        if not isinstance(data, dict):
            raise ValueError("输入根节点必须是 JSON 对象")
        required = {"materials", "plies", "reference_load"}
        missing, extra = required - data.keys(), data.keys() - required - {"title"}
        if missing or extra:
            raise ValueError(f"输入字段错误：缺少 {sorted(missing)}，未知 {sorted(extra)}")
        if not isinstance(data["materials"], dict) or not isinstance(data["plies"], list):
            raise ValueError("materials 必须是对象，plies 必须是数组")
        if not isinstance(data["reference_load"], list):
            raise ValueError("reference_load 必须是数组")
        materials = {}
        for name, values in data["materials"].items():
            try:
                materials[name] = Material(**values)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"材料 {name!r}：{exc}") from exc
        plies = []
        for i, values in enumerate(data["plies"], 1):
            try:
                plies.append(Ply(**values))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"第 {i} 层：{exc}") from exc
        return cls(materials, tuple(plies), tuple(data["reference_load"]), data.get("title", "层合板渐进失效分析"))

    @classmethod
    def from_json(cls, path: str | Path) -> Laminate:
        with Path(path).open(encoding="utf-8-sig") as handle:
            return cls.from_dict(json.load(handle))


def transformation(angle: float) -> np.ndarray:
    """Global to material stress transformation (engineering shear convention)."""
    radians = math.radians(angle % 360)
    m, n = math.cos(radians), math.sin(radians)
    return np.array([[m*m, n*n, 2*m*n], [n*n, m*m, -2*m*n],
                     [-m*n, m*n, m*m-n*n]])


def degraded_stiffness(original: np.ndarray, state: str) -> np.ndarray:
    q = np.array(original, copy=True)
    if state == "fully_failed":
        q[:] = 0
    elif state == "transverse_failed":
        q[:] = 0
        q[0, 0] = original[0, 0]
    elif state == "shear_failed":
        q[2, :] = q[:, 2] = 0
    elif state != "intact":
        raise ValueError(f"未知损伤状态：{state}")
    return q


def assemble_abd(model: Laminate, states: tuple[str, ...] | None = None) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return ABD and per-ply transformed reduced stiffness matrices."""
    states = ("intact",) * len(model.plies) if states is None else states
    if len(states) != len(model.plies):
        raise ValueError("损伤状态数量必须等于铺层数")
    a, b, d = (np.zeros((3, 3)) for _ in range(3))
    qbars = []
    z = model.interfaces
    for i, (ply, state) in enumerate(zip(model.plies, states)):
        q = degraded_stiffness(model.materials[ply.material].stiffness(), state)
        inverse_t = np.linalg.solve(transformation(ply.angle), np.eye(3))
        qbar = inverse_t @ q @ inverse_t.T
        qbar = (qbar + qbar.T) / 2
        qbars.append(qbar)
        z_top, z_bottom = z[i:i+2]
        # Factored differences avoid subtracting nearly equal powers for thin plies.
        a += qbar * ply.thickness
        b += qbar * ply.thickness * (z_bottom + z_top) / 2
        d += qbar * ply.thickness * (z_bottom*z_bottom + z_bottom*z_top + z_top*z_top) / 3
    return np.block([[a, b], [b, d]]), tuple(qbars)


class StiffnessLoss(ValueError):
    """The thickness-scaled ABD no longer supports a unique stable solution."""


class UnsupportedLoad(StiffnessLoss):
    """The applied load has a component along a zero-stiffness mode."""


def solve_response(abd: np.ndarray, load: tuple[float, ...] | np.ndarray, thickness: float,
                   *, allow_singular: bool = False) -> np.ndarray:
    """Solve equilibrium, optionally on the surviving stiffness subspace.

    In singular states, zero-stiffness coordinates of [epsilon, h*kappa] are
    set to zero. This selects a minimum-norm representative, not a unique
    physical deformation. An incompatible load is never projected away.
    """
    scale = np.array([1., 1., 1., 1/thickness, 1/thickness, 1/thickness])
    scaled = abd * np.outer(scale, scale) / thickness
    if not np.all(np.isfinite(scaled)):
        raise ValueError("ABD 超出有限数值范围，请检查输入量级")
    eigenvalues, eigenvectors = np.linalg.eigh(scaled)
    largest, smallest = eigenvalues[-1], eigenvalues[0]
    threshold = STIFFNESS_RTOL * max(largest, 0.)
    if smallest < -threshold or (not allow_singular and (largest <= 0 or smallest <= threshold)):
        raise StiffnessLoss(f"厚度缩放后的 ABD 奇异、非正定或数值退化；最小/最大特征值比 "
                            f"{smallest / largest if largest > 0 else 0:.6g}，阈值 {STIFFNESS_RTOL:g}")
    rhs = np.asarray(load) * scale / thickness
    try:
        if allow_singular and smallest <= threshold:
            surviving = eigenvalues > threshold
            coordinates = eigenvectors.T @ rhs
            if np.linalg.norm(coordinates[~surviving]) > EQUILIBRIUM_RTOL * np.linalg.norm(rhs):
                raise UnsupportedLoad("仍有铺层未完全失效，但当前载荷在零刚度方向存在分量，无法满足静力平衡。")
            solution = eigenvectors[:, surviving] @ (coordinates[surviving] / eigenvalues[surviving])
            if np.linalg.norm(scaled @ solution - rhs) > EQUILIBRIUM_RTOL * np.linalg.norm(rhs):
                raise StiffnessLoss("剩余刚度子空间求解未通过载荷平衡残差检查。")
            response = scale * solution
        else:
            response = scale * np.linalg.solve(scaled, rhs)
    except np.linalg.LinAlgError as exc:
        raise StiffnessLoss("ABD 无法求解") from exc
    if not np.all(np.isfinite(response)):
        raise ValueError("应变或曲率超出有限数值范围")
    return response


def surface_stresses(model: Laminate, qbars: tuple[np.ndarray, ...], response: np.ndarray,
                     states: tuple[str, ...] | None = None) -> np.ndarray:
    """Shape (number of plies, top/bottom, sigma1/sigma2/tau12)."""
    result = np.empty((len(model.plies), 2, 3))
    z = model.interfaces
    for i, ply in enumerate(model.plies):
        for j in range(2):
            result[i, j] = transformation(ply.angle) @ qbars[i] @ (response[:3] + z[i+j] * response[3:])
        # Remove roundoff-only components in directions with exactly zero stiffness.
        if states is not None:
            if states[i] == "fully_failed":
                result[i] = 0
            elif states[i] == "transverse_failed":
                result[i, :, 1:] = 0
            elif states[i] == "shear_failed":
                result[i, :, 2] = 0
    if not np.all(np.isfinite(result)):
        raise ValueError("应力超出有限数值范围")
    return result


def hoffman_coefficients(stress: np.ndarray, material: Material) -> tuple[float, float]:
    """FI(lambda * stress) = a * lambda**2 + b * lambda."""
    s1, s2, t = map(float, stress)
    m = material
    a = (s1/m.Xt)*(s1/m.Xc) - (s1/m.Xt)*(s2/m.Xc) + (s2/m.Yt)*(s2/m.Yc) + (t/m.S)**2
    b = s1 * (1/m.Xt - 1/m.Xc) + s2 * (1/m.Yt - 1/m.Yc)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Hoffman 系数超出有限数值范围")
    return a, b


def hoffman_index(stress: np.ndarray, material: Material) -> float:
    a, b = hoffman_coefficients(stress, material)
    return a + b


def positive_failure_roots(a: float, b: float) -> tuple[float, ...]:
    """Stable positive roots of a*x*x+b*x=1, including nonconvex cases."""
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("失效方程系数必须有限")
    if a == 0:
        return (1/b,) if b > 0 and math.isfinite(1/b) else ()
    scale = max(math.sqrt(abs(a)), abs(b))
    aa, bb = (a / scale) / scale, b / scale
    if aa == 0:  # Only a root representable at this numerical scale is relevant.
        return (1/b,) if b > 0 and math.isfinite(1/b) else ()
    discriminant = bb*bb + 4*aa
    tolerance = 32*np.finfo(float).eps * (bb*bb + abs(4*aa))
    if discriminant < -tolerance:
        return ()
    q = -0.5 * (bb + math.copysign(math.sqrt(max(0., discriminant)), bb))
    candidates = ((q / aa) / scale, (-1 / q) / scale)
    return tuple(sorted({r for r in candidates if r > 0 and math.isfinite(r)}))


def failure_mode(stress: np.ndarray, material: Material, state: str) -> tuple[str, tuple[float, ...]]:
    s1, s2, shear = map(float, stress)
    ratios = (abs(s1)/(material.Xt if s1 >= 0 else material.Xc),
              abs(s2)/(material.Yt if s2 >= 0 else material.Yc), abs(shear)/material.S)
    active = {"intact": (0, 1, 2), "shear_failed": (0, 1), "transverse_failed": (0,)}.get(state, ())
    if not active:
        raise ValueError("完全失效层不能继续退化")
    exceeded = [i for i in active if ratios[i] >= 1 - FAILURE_TOL]
    if exceeded:
        selected = exceeded[0]
    else:
        maximum = max(ratios[i] for i in active)
        selected = next(i for i in active if ratios[i] >= maximum - FAILURE_TOL * max(1., maximum))
    return MODES[selected], ratios


@dataclass(frozen=True)
class FailureEvent:
    load_factor: float
    load_step: int
    cascade_step: int
    ply: int  # One-based, top to bottom.
    surface: str
    z: float
    mode: str
    stress: tuple[float, ...]
    failure_index: float
    utilization: tuple[float, ...]
    state_before: str
    state_after: str
    history_index: int


@dataclass(frozen=True)
class HistoryPoint:
    load_factor: float
    response: tuple[float, ...]
    states: tuple[str, ...]
    stage: str
    load_step: int
    cascade_step: int
    event_indices: tuple[int, ...]  # Zero-based indices into AnalysisResult.events.
    stiffness_rank: int = 6  # Below six: a minimum-norm equilibrium representative.


@dataclass(frozen=True)
class AnalysisResult:
    model: Laminate
    events: tuple[FailureEvent, ...]
    history: tuple[HistoryPoint, ...]
    status: str
    reason: str
    terminal_load_factor: float | None
    final_states: tuple[str, ...]
    initial_abd: np.ndarray

    @property
    def first_failure_load_factor(self) -> float | None:
        return self.events[0].load_factor if self.events else None

    @property
    def ultimate_load_factor(self) -> float | None:
        return self.terminal_load_factor if self.status == "all_plies_failed" else None


def analyze(model: Laminate) -> AnalysisResult:
    """Follow loading until every ply is fully failed (all local Q are zero).

    All failing plies are degraded together. At a fixed load, the same solver
    repeats until redistribution is stable, without advancing the load.
    """
    states = ("intact",) * len(model.plies)
    initial_abd, _ = assemble_abd(model, states)
    events: list[FailureEvent] = []
    history: list[HistoryPoint] = []
    factor, load_step, cascade = 0., 0, 0
    z = model.interfaces
    state_for_mode = {"fiber": "fully_failed", "transverse": "transverse_failed", "shear": "shear_failed"}
    while True:
        if all(state == "fully_failed" for state in states):
            return AnalysisResult(model, tuple(events), tuple(history), "all_plies_failed",
                                  "所有单层板均已完全失效（各层材料坐标系 Q 均为零）。",
                                  factor, states, initial_abd)
        abd, qbars = assemble_abd(model, states)
        try:
            unit_response = solve_response(abd, model.reference_load, model.thickness,
                                           allow_singular=bool(events))
        except UnsupportedLoad as exc:
            return AnalysisResult(model, tuple(events), tuple(history), "load_unsupported", str(exc),
                                  factor, states, initial_abd)
        except StiffnessLoss as exc:
            if not events:
                raise ValueError(f"初始层合板无有效整体刚度：{exc}") from exc
            return AnalysisResult(model, tuple(events), tuple(history), "numerical_failure", str(exc),
                                  factor, states, initial_abd)
        scale = np.array([1., 1., 1., 1/model.thickness, 1/model.thickness, 1/model.thickness])
        eigenvalues = np.linalg.eigvalsh(abd * np.outer(scale, scale) / model.thickness)
        rank = int(np.count_nonzero(eigenvalues > STIFFNESS_RTOL * eigenvalues[-1]))
        unit_stress = surface_stresses(model, qbars, unit_response, states)
        failing = []
        for i, ply in enumerate(model.plies):
            if states[i] == "fully_failed":
                continue
            material = model.materials[ply.material]
            candidates = []
            for side in range(2):
                stress = unit_stress[i, side] * factor
                index = hoffman_index(stress, material)
                if index >= 1 - FAILURE_TOL:
                    mode, ratios = failure_mode(stress, material, states[i])
                    candidates.append((MODES.index(mode), -index, side, mode, ratios, stress))
            if candidates:
                # If both surfaces fail, apply the more severe mode to the whole ply.
                chosen = min(candidates, key=lambda c: c[:3])
                failing.append((i, chosen))
        if factor and cascade == 0 and not failing:
            raise ArithmeticError("失效根未通过 Hoffman 复核；请检查输入数值量级")
        first_event = len(events)
        next_states = list(states)
        for i, (_, negative_index, side, mode, ratios, stress) in failing:
            after = state_for_mode[mode]
            if STATES.index(after) <= STATES.index(states[i]):
                raise ArithmeticError("退化未推进，终止以防止重复损伤")
            events.append(FailureEvent(factor, load_step, cascade, i+1,
                                       "top" if side == 0 else "bottom", float(z[i+side]), mode,
                                       tuple(map(float, stress)), -negative_index, ratios,
                                       states[i], after, len(history)))
            next_states[i] = after
        stage = "initial" if factor == 0 else ("before_failure" if failing else "redistributed")
        history.append(HistoryPoint(factor, tuple(map(float, unit_response * factor)), states,
                                    stage, load_step, cascade, tuple(range(first_event, len(events))), rank))
        if failing:
            if len(events) > 3 * len(model.plies):
                raise ArithmeticError("损伤事件数超过每层最多三次的理论上限")
            states = tuple(next_states)
            cascade += 1
            continue
        roots = []
        for i, ply in enumerate(model.plies):
            if states[i] == "fully_failed":
                continue
            for side in range(2):
                coefficients = hoffman_coefficients(unit_stress[i, side], model.materials[ply.material])
                roots.extend(r for r in positive_failure_roots(*coefficients) if r > factor)
        if not roots:
            return AnalysisResult(model, tuple(events), tuple(history), "no_finite_failure",
                                  "本模型未得到有限失效载荷：当前损伤状态下无后续有效正失效根。",
                                  None, states, initial_abd)
        factor = min(roots)
        load_step += 1
        cascade = 0
