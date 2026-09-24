"""盖斯定律（Hess's law）反应焓推导的精确有理数求解。

给定一条**目标反应**与至多 6 条**已知反应**（均已按最小互质正整数唯一
配平，附带以 kJ 计的反应焓），把每条反应表示成“生成物为正、反应物为
负”的计量向量，在有理数域上求有符号线性组合

    Σ λⱼ·vⱼ = v_target ， λⱼ ∈ ℚ（允许反向即 λ<0、允许分数倍）

使合并抵消后的净计量向量恰好等于目标向量；推导出的目标反应焓为

    ΔH_target = Σ λⱼ·ΔHⱼ 。

判定规则：

* 目标向量不在已知向量张成的列空间内 → :class:`HessError`
  （``HESS_NOT_DERIVABLE``，不可推导）；
* 列空间存在零空间向量 z（Σzⱼvⱼ=0）但 ΣzⱼΔHⱼ ≠ 0 →
  ``HESS_AMBIGUOUS``（多种组合导出不同反应焓，数据不自洽）；
* 全部零空间方向的焓组合均为 0 时，任意解导出的反应焓相同，正常返回；
  此时在全部可行解中选取**用到已知反应最少**的规范解（同支撑集大小取
  下标字典序最小），未使用的反应倍数为 0，保证输出确定。

全程使用 :class:`fractions.Fraction`，不经过 ``float``。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from itertools import combinations
from math import gcd, lcm

from .balancer import Species, _null_space, _rref
from .cache import species_key
from .stoichiometry import MAX_ABS_EXPONENT, decimal_to_fraction


class HessError(Exception):
    """反应焓无法推导时抛出，携带机器可读的错误码。"""

    def __init__(self, code: str, message: str, **details: object) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


@dataclass(frozen=True)
class BalancedReaction:
    """一条已唯一配平的反应：两侧物种与最小互质正整数系数。"""

    reactants: tuple[Species, ...]
    products: tuple[Species, ...]
    coefficients: dict[str, int]

    def vector(self, species_keys: list[str]) -> list[Fraction]:
        """按给定物种顺序生成“生成物为正、反应物为负”的计量向量。"""
        signed: dict[str, Fraction] = {}
        for sp in self.reactants:
            signed[species_key(sp)] = Fraction(-self.coefficients[species_key(sp)])
        for sp in self.products:
            signed[species_key(sp)] = Fraction(self.coefficients[species_key(sp)])
        return [signed.get(key, Fraction(0)) for key in species_keys]


@dataclass(frozen=True)
class HessKnownReaction(BalancedReaction):
    """一条已知反应：唯一配平的反应式 + 以 kJ 计的精确有理数反应焓。"""

    enthalpy: Fraction = Fraction(0)


@dataclass(frozen=True)
class HessSolution:
    """盖斯推导结果。

    ``multipliers`` 与已知反应输入顺序一一对应（未使用者为 0）；
    ``net_coefficients`` 覆盖目标与全部已知反应涉及的并集物种
    （生成物侧为正、反应物侧为负，抵消掉的中间体为 0）。
    """

    multipliers: tuple[Fraction, ...]
    target_enthalpy: Fraction
    net_coefficients: dict[str, Fraction]


def parse_enthalpy(text: str) -> Fraction:
    """把以 kJ 计的（可负）十进制字符串无损解析为 :class:`Fraction`。

    拒绝 NaN / 无穷、非数字字符串与过大数量级；允许科学计数法。
    """
    try:
        value = Decimal(text.strip())
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError(f"反应焓不是合法的十进制数字：{text!r}") from exc
    if not value.is_finite():
        raise ValueError("反应焓不能是 NaN 或无穷大")
    if value.adjusted() > MAX_ABS_EXPONENT or value.as_tuple().exponent < -MAX_ABS_EXPONENT:
        raise ValueError(f"反应焓的数量级超出允许范围（±1e{MAX_ABS_EXPONENT}）")
    return decimal_to_fraction(value)


def solve_hess(
    target: BalancedReaction, known: list[HessKnownReaction]
) -> HessSolution:
    """求已知反应的有符号有理线性组合等于目标反应，并推导目标反应焓。"""
    # 目标与已知反应的物种并集（规范键），矩阵行顺序由此固定
    union_species: dict[str, Species] = {}
    for rxn in (target, *known):
        for sp in (*rxn.reactants, *rxn.products):
            union_species[species_key(sp)] = sp
    species_order = sorted(union_species)

    # 已知反应作为列：columns[i][j] 为第 j 条已知反应在第 i 个物种上的分量
    n = len(known)
    columns: list[list[Fraction]] = [
        [known[j].vector(species_order)[i] for j in range(n)]
        for i in range(len(species_order))
    ]
    target_vec = target.vector(species_order)

    _check_in_span(columns, target_vec, union_species)
    multipliers = _canonical_multipliers(columns, target_vec, n)
    _check_enthalpy_consistency(columns, [rxn.enthalpy for rxn in known])

    multipliers_tuple = tuple(multipliers)
    target_enthalpy = sum(
        (lam * rxn.enthalpy for lam, rxn in zip(multipliers, known)),
        Fraction(0),
    )
    # 合并抵消后的逐物种净系数（= 目标向量；已知反应独有物种抵消为 0）
    net: dict[str, Fraction] = {}
    for i, key in enumerate(species_order):
        net[key] = sum(
            (lam * columns[i][j] for j, lam in enumerate(multipliers)),
            Fraction(0),
        )
        if net[key] != target_vec[i]:
            raise HessError(  # 理论上不可达：已通过列空间与一致性检查
                "HESS_NOT_DERIVABLE",
                f"内部校验失败：物种 {union_species[key].normalized} 的净系数"
                f"{net[key]} 与目标系数 {target_vec[i]} 不一致",
            )
    return HessSolution(multipliers_tuple, target_enthalpy, net)


# ---------------------------------------------------------------- 列空间判定
def _augmented_rref(
    columns: list[list[Fraction]],
    target: list[Fraction],
    col_indices: list[int] | None = None,
) -> tuple[list[list[Fraction]], list[int]]:
    """对 [A[:,col_indices] | b] 做 RREF（col_indices 缺省取全部列）。"""
    if col_indices is None:
        col_indices = list(range(len(columns[0])))
    aug = [
        [row[j] for j in col_indices] + [target[i]]
        for i, row in enumerate(columns)
    ]
    return _rref(aug)


def _is_consistent(rref: list[list[Fraction]]) -> bool:
    """RREF 后出现“系数全 0、右端非 0”的行即方程组无解。"""
    return not any(
        all(value == 0 for value in row[:-1]) and row[-1] != 0 for row in rref
    )


def _check_in_span(
    columns: list[list[Fraction]],
    target: list[Fraction],
    union_species: dict[str, Species],
) -> None:
    """目标向量不在已知列张成空间内时报 HESS_NOT_DERIVABLE。"""
    species_order = sorted(union_species)
    rref, _pivots = _augmented_rref(columns, target)
    if _is_consistent(rref):
        return

    # 诊断：目标侧系数非零、但在全部已知反应中都不出现的物种必然无法凑出
    unreachable = [
        union_species[species_order[i]].normalized
        for i, value in enumerate(target)
        if value != 0 and all(row == 0 for row in columns[i])
    ]
    details: dict[str, object] = {}
    if unreachable:
        details["unreachable_species"] = sorted(unreachable)
    raise HessError(
        "HESS_NOT_DERIVABLE",
        "目标反应不在已知反应的张成空间内：不存在已知反应的有符号有理"
        "线性组合（允许反向与分数倍使用）使其计量向量等于目标反应，"
        "无法用盖斯定律推导目标反应焓；请补充参与目标反应物种的已知反应。",
        **details,
    )


# ---------------------------------------------------------------- 规范解
def _solve_unique_subset(
    columns: list[list[Fraction]],
    target: list[Fraction],
    col_indices: tuple[int, ...],
) -> list[Fraction] | None:
    """求 A[:,col_indices] x = b 的唯一解；无解或不唯一（列不满秩）时返回 None。"""
    rref, pivots = _augmented_rref(columns, target, list(col_indices))
    if not _is_consistent(rref):
        return None
    if len(pivots) < len(col_indices):
        return None  # 列线性相关 -> 无穷多解，不作为最小支撑集
    # RREF 主元列下标是子矩阵的局部位置，映射回原已知反应列号
    return [rref[row_idx][-1] for row_idx, _col in enumerate(pivots)]


def _canonical_multipliers(
    columns: list[list[Fraction]], target: list[Fraction], n: int
) -> list[Fraction]:
    """在全部可行解中选支撑集最小、同大小按下标字典序最小的规范解。"""
    for size in range(1, n + 1):
        for subset in combinations(range(n), size):
            values = _solve_unique_subset(columns, target, subset)
            if values is not None:
                full = [Fraction(0)] * n
                for col, value in zip(subset, values):
                    full[col] = value
                return full
    raise HessError(  # 理论上不可达：已通过列空间一致性检查
        "HESS_NOT_DERIVABLE",
        "内部错误：目标向量在列空间内但找不到最小支撑解",
    )


# ---------------------------------------------------------------- 焓自洽性
def _primitive_integers(vec: list[Fraction]) -> list[int]:
    """有理向量按分母 LCM 放大、再除以 gcd，化为最简整数向量。"""
    scale = lcm(*(value.denominator for value in vec)) if vec else 1
    scaled = [int(value * scale) for value in vec]
    common = 0
    for value in scaled:
        common = gcd(common, abs(value))
    if common == 0:
        return scaled
    return [value // common for value in scaled]


def _check_enthalpy_consistency(
    columns: list[list[Fraction]], enthalpies: list[Fraction]
) -> None:
    """列空间的每个零方向都必须满足 ΣzⱼΔHⱼ = 0，否则反应焓有歧义。"""
    # A 的零空间基：Σ zⱼvⱼ = 0 的全部组合方向
    basis = _null_space([row[:] for row in columns])
    conflicts: list[dict[str, object]] = []
    for direction in basis:
        delta = sum(
            (z * h for z, h in zip(direction, enthalpies)), Fraction(0)
        )
        if delta != 0:
            primitive = _primitive_integers(direction)
            conflicts.append(
                {
                    "multipliers": [str(value) for value in primitive],
                    "enthalpy_difference_kj": str(delta),
                }
            )
    if conflicts:
        raise HessError(
            "HESS_AMBIGUOUS",
            "已知反应存在多种线性组合都能得到目标反应，但它们导出的目标"
            "反应焓不相等（已知反应数据不自洽：某些已知反应的零净计量"
            "组合具有非零反应焓）；请核对已知反应焓后重试。",
            conflicts=conflicts,
        )
