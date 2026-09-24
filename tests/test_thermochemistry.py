"""盖斯定律核心求解器 app.thermochemistry 的单元测试。

直接在精确有理数层面验证：向量符号约定、有符号/分数线性组合、最小支撑
规范解的确定性、焓自洽性检查（零空间方向）、不可推导与歧义错误码，
以及 parse_enthalpy 的十进制无损解析。
"""

import pytest

from app.balancer import BalanceError, solve_reaction
from app.cache import species_key
from app.thermochemistry import (
    BalancedReaction,
    HessError,
    HessKnownReaction,
    parse_enthalpy,
    solve_hess,
)
from fractions import Fraction


def _solved_parts(reactants, products):
    solved = solve_reaction(reactants, products)
    coefficients = {
        species_key(sp): coef
        for sp, coef in zip(solved.all_species, solved.coefficients)
    }
    return solved, coefficients


def known(reactants, products, enthalpy):
    """通过既有配平核心构造一条 HessKnownReaction。"""
    solved, coefficients = _solved_parts(reactants, products)
    return HessKnownReaction(
        reactants=solved.reactants,
        products=solved.products,
        coefficients=coefficients,
        enthalpy=parse_enthalpy(enthalpy),
    )


def target_reaction(reactants, products):
    solved, coefficients = _solved_parts(reactants, products)
    return BalancedReaction(
        reactants=solved.reactants,
        products=solved.products,
        coefficients=coefficients,
    )


def test_vector_sign_convention_products_positive():
    rxn = known(["H2", "O2"], ["H2O"], "-571.6")
    keys = [species_key(sp) for sp in (*rxn.reactants, *rxn.products)]
    vec = rxn.vector(keys)
    # 生成物为正、反应物为负，与配平整数系数一致
    values = {sp.normalized: v for sp, v in zip((*rxn.reactants, *rxn.products), vec)}
    assert values == {"H2": Fraction(-2), "O2": Fraction(-1), "H2O": Fraction(2)}


def test_signed_and_fractional_combination():
    # S + O2 -> SO2 = 1/2*(2S+3O2->2SO3) - 1/2*(2SO2+O2->2SO3)
    target = target_reaction(["S", "O2"], ["SO2"])
    b = known(["SO2", "O2"], ["SO3"], "-197.0")
    c = known(["S", "O2"], ["SO3"], "-790.6")
    result = solve_hess(target, [b, c])
    assert result.multipliers == (Fraction(-1, 2), Fraction(1, 2))
    assert result.target_enthalpy == Fraction(-2968, 10)
    # 中间产物 SO3 抵消为 0，目标物种净系数与目标向量一致
    net = result.net_coefficients
    expected = {"SO3": 0, "SO2": 1, "S": -1, "O2": -1}
    for sp in (*b.reactants, *b.products, *c.reactants, *c.products):
        assert net[species_key(sp)] == Fraction(expected[sp.normalized])


def test_canonical_solution_minimal_support_deterministic():
    # 三条已知：A、B 可组合出目标，C 直接等于目标；焓自洽时规范解只用 C
    target = target_reaction(["S", "O2"], ["SO3"])
    a = known(["S", "O2"], ["SO2"], "-296.8")
    b = known(["SO2", "O2"], ["SO3"], "-197.0")
    c = known(["S", "O2"], ["SO3"], "-790.6")
    result = solve_hess(target, [a, b, c])
    assert result.multipliers == (Fraction(0), Fraction(0), Fraction(1))
    # 换序后最小支撑仍取“下标字典序最小”的直接命中列
    result_rev = solve_hess(target, [c, a, b])
    assert result_rev.multipliers == (Fraction(1), Fraction(0), Fraction(0))
    assert result.target_enthalpy == result_rev.target_enthalpy


def test_not_derivable_raises():
    target = target_reaction(["C", "H2"], ["CH4"])
    a = known(["C", "O2"], ["CO2"], "-393.5")
    with pytest.raises(HessError) as exc:
        solve_hess(target, [a])
    assert exc.value.code == "HESS_NOT_DERIVABLE"
    assert set(exc.value.details["unreachable_species"]) == {"CH4", "H2"}


def test_ambiguous_raises_with_conflict():
    target = target_reaction(["S", "O2"], ["SO3"])
    a = known(["S", "O2"], ["SO2"], "-296.8")
    b = known(["SO2", "O2"], ["SO3"], "-197.0")
    c = known(["S", "O2"], ["SO3"], "-790.0")  # 自洽应为 -790.6
    with pytest.raises(HessError) as exc:
        solve_hess(target, [a, b, c])
    assert exc.value.code == "HESS_AMBIGUOUS"
    conflict = exc.value.details["conflicts"][0]
    assert conflict["multipliers"] == ["-2", "-1", "1"]
    assert Fraction(conflict["enthalpy_difference_kj"]) == Fraction(3, 5)


def test_same_enthalpy_dependent_reactions_ok():
    # 同一反应给出相同焓：零组合焓差为 0，推导正常
    target = target_reaction(["S", "O2"], ["SO2"])
    a = known(["S", "O2"], ["SO2"], "-296.8")
    b = known(["S", "O2"], ["SO2"], "-296.8")
    result = solve_hess(target, [a, b])
    assert result.multipliers == (Fraction(1), Fraction(0))
    assert result.target_enthalpy == Fraction(-2968, 10)


def test_reverse_usage_negative_multiplier():
    # 2C + O2 -> 2CO = 2A - B（B: 2CO+O2->2CO2）
    target = target_reaction(["C", "O2"], ["CO"])
    a = known(["C", "O2"], ["CO2"], "-393.5")
    b = known(["CO", "O2"], ["CO2"], "-566.0")
    result = solve_hess(target, [a, b])
    assert result.multipliers == (Fraction(2), Fraction(-1))
    assert result.target_enthalpy == Fraction(-2210, 10)


@pytest.mark.parametrize("text,expected", [
    ("-285.8", Fraction(-2858, 10)),
    ("+90.25", Fraction(9025, 100)),
    ("0", Fraction(0)),
    ("  -1.5e2 ", Fraction(-150)),
    ("1/3", None),  # 分数写法不接受
])
def test_parse_enthalpy_decimal(text, expected):
    if expected is None:
        with pytest.raises(ValueError):
            parse_enthalpy(text)
    else:
        assert parse_enthalpy(text) == expected


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity", "abc", ""])
def test_parse_enthalpy_rejects(text):
    with pytest.raises(ValueError):
        parse_enthalpy(text)


def test_balance_failures_propagate_from_core():
    # 配平核心的错误（如不唯一配平）不会被盖斯层吞掉
    with pytest.raises(BalanceError):
        known(["C", "O2"], ["CO", "CO2"], "0")
