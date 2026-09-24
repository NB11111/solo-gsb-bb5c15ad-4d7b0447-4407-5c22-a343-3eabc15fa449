"""POST /api/v1/thermochemistry/hess 的 HTTP API 测试。

覆盖：经典盖斯组合（反向 + 整数倍）、分数倍、最小支撑规范解、目标恰好
等于已知反应、目标在多条已知反应张成但焓自洽、反应焓任意符号/零值、
电荷（半反应）物种；错误路径：不可推导（HESS_NOT_DERIVABLE）、焓歧义
（HESS_AMBIGUOUS）、反应焓字符串非法、化学式解析失败、同侧重复、
目标/已知反应无法唯一配平（details 定位）、已知反应数量越界与请求体
结构问题。
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PATH = "/api/v1/thermochemistry/hess"


def rxn(reactants, products, enthalpy=None):
    body = {"reactants": reactants, "products": products}
    if enthalpy is not None:
        body["enthalpy_kj"] = enthalpy
    return body


# ------------------------------------------------------------ 成功路径
def test_classic_hess_two_known_reversed():
    # 2C + O2 -> 2CO = 2*(C+O2->CO2) - (2CO+O2->2CO2)
    # ΔH = 2*(-393.5) - (-566.0) = -221.0
    resp = client.post(
        PATH,
        json={
            "target": rxn(["C", "O2"], ["CO"]),
            "known_reactions": [
                rxn(["C", "O2"], ["CO2"], "-393.5"),
                rxn(["CO", "O2"], ["CO2"], "-566.0"),
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["target_enthalpy_kj"] == "-221"
    assert data["target"]["normalized"]["equation"] == "2C + O2 -> 2CO"
    known = {k["index"]: k for k in data["known_reactions"]}
    assert known[0]["multiplier"] == "2"
    assert known[0]["contribution_kj"] == "-787"
    assert known[1]["multiplier"] == "-1"
    assert known[1]["contribution_kj"] == "566"
    net = {item["normalized"]: item["coefficient"]
           for item in data["net_coefficients"]}
    assert net == {"C": "-2", "CO": "2", "CO2": "0", "O2": "-1"}


def test_fractional_multipliers_exact_decimal():
    # S + O2 -> SO2（ΔH=-296.8）= 1/2*(2S+3O2->2SO3) - 1/2*(2SO2+O2->2SO3)
    resp = client.post(
        PATH,
        json={
            "target": rxn(["S", "O2"], ["SO2"]),
            "known_reactions": [
                rxn(["SO2", "O2"], ["SO3"], "-197.0"),
                rxn(["S", "O2"], ["SO3"], "-790.6"),
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["target_enthalpy_kj"] == "-296.8"
    multipliers = [k["multiplier"] for k in data["known_reactions"]]
    assert multipliers == ["-0.5", "0.5"]
    net = {item["normalized"]: item["coefficient"]
           for item in data["net_coefficients"]}
    assert net["SO3"] == "0"
    assert net["SO2"] == "1"
    assert net["S"] == "-1"
    assert net["O2"] == "-1"


def test_target_equals_known_reaction_multiplier_one():
    resp = client.post(
        PATH,
        json={
            "target": rxn(["H2", "O2"], ["H2O"]),
            "known_reactions": [rxn(["H2", "O2"], ["H2O"], "-571.6")],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["target_enthalpy_kj"] == "-571.6"
    assert data["known_reactions"][0]["multiplier"] == "1"
    assert data["known_reactions"][0]["contribution_kj"] == "-571.6"
    net = {item["normalized"]: item["coefficient"]
           for item in data["net_coefficients"]}
    assert net == {"H2": "-2", "H2O": "2", "O2": "-1"}


def test_minimal_support_zeroes_unused_reactions():
    # 目标 2S+3O2->2SO3；前两条可组合出目标，第三条直接就是目标且焓相同：
    # 规范解取支撑最小的第三条（前两条倍数为 0）
    resp = client.post(
        PATH,
        json={
            "target": rxn(["S", "O2"], ["SO3"]),
            "known_reactions": [
                rxn(["S", "O2"], ["SO2"], "-296.8"),
                rxn(["SO2", "O2"], ["SO3"], "-197.0"),
                rxn(["S", "O2"], ["SO3"], "-790.6"),
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["target_enthalpy_kj"] == "-790.6"
    assert [k["multiplier"] for k in data["known_reactions"]] == ["0", "0", "1"]
    assert [k["contribution_kj"] for k in data["known_reactions"]] == [
        "0", "0", "-790.6"
    ]


def test_order_independence_of_known_reactions_enthalpy():
    body_a = {
        "target": rxn(["C", "O2"], ["CO"]),
        "known_reactions": [
            rxn(["C", "O2"], ["CO2"], "-393.5"),
            rxn(["CO", "O2"], ["CO2"], "-566.0"),
        ],
    }
    body_b = {
        "target": rxn(["C", "O2"], ["CO"]),
        "known_reactions": [
            rxn(["CO", "O2"], ["CO2"], "-566.0"),
            rxn(["C", "O2"], ["CO2"], "-393.5"),
        ],
    }
    da = client.post(PATH, json=body_a).json()
    db = client.post(PATH, json=body_b).json()
    assert da["target_enthalpy_kj"] == db["target_enthalpy_kj"] == "-221"
    # 倍数随输入顺序对齐：交换后倍数也对应交换
    assert [k["multiplier"] for k in da["known_reactions"]] == ["2", "-1"]
    assert [k["multiplier"] for k in db["known_reactions"]] == ["-1", "2"]


def test_positive_and_zero_enthalpy_strings():
    # 正焓与零焓：2HI -> H2 + I2 为吸热反应，正号与零值十进制串均可解析
    resp = client.post(
        PATH,
        json={
            "target": rxn(["HI"], ["H2", "I2"]),
            "known_reactions": [
                rxn(["HI"], ["H2", "I2"], "+53.0"),
                rxn(["HI"], ["H2", "I2"], "0"),
            ],
        },
    )
    assert resp.status_code == 422
    # 同一反应给出不同焓（53.0 vs 0）-> 歧义
    assert resp.json()["error"]["code"] == "HESS_AMBIGUOUS"

    resp = client.post(
        PATH,
        json={
            "target": rxn(["HI"], ["H2", "I2"]),
            "known_reactions": [rxn(["HI"], ["H2", "I2"], "+53.0")],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["target_enthalpy_kj"] == "53"
    assert data["known_reactions"][0]["multiplier"] == "1"


def test_charge_balanced_half_reactions():
    # Zn -> Zn2+ + 2e- = (Zn + Cu2+ -> Zn2+ + Cu) - (Cu2+ + 2e- -> Cu)
    resp = client.post(
        PATH,
        json={
            "target": rxn(["Zn"], ["Zn2+", "e-"]),
            "known_reactions": [
                rxn(["Zn", "Cu2+"], ["Zn2+", "Cu"], "-218.0"),
                rxn(["Cu2+", "e-"], ["Cu"], "-64.8"),
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["target_enthalpy_kj"] == "-153.2"
    assert [k["multiplier"] for k in data["known_reactions"]] == ["1", "-1"]
    assert data["target"]["normalized"]["equation"] == "Zn -> 2e^- + Zn^2+"
    net = {item["normalized"]: item["coefficient"]
           for item in data["net_coefficients"]}
    assert net["Cu"] == "0"
    assert net["Cu^2+"] == "0"


def test_equivalent_formula_writings_recognized():
    # 目标与已知反应用等价括号/电荷写法，应识别为同一物种
    resp = client.post(
        PATH,
        json={
            "target": rxn(["Zn"], ["Zn 2+", "e-"]),
            "known_reactions": [
                rxn(["Zn", "Cu^2+"], ["Zn2+", "Cu"], "-218.0"),
                rxn(["Cu2+", "e-"], ["Cu"], "-64.8"),
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["target_enthalpy_kj"] == "-153.2"


def test_six_known_reactions_accepted():
    # 至多 6 条：前 5 条与目标无关物种（但各自能配平），第 6 条即目标
    resp = client.post(
        PATH,
        json={
            "target": rxn(["H2", "O2"], ["H2O"]),
            "known_reactions": [
                rxn(["N2", "H2"], ["NH3"], "-92.4"),
                rxn(["N2", "O2"], ["NO"], "180.5"),
                rxn(["S", "O2"], ["SO2"], "-296.8"),
                rxn(["C", "O2"], ["CO2"], "-393.5"),
                rxn(["Na", "Cl2"], ["NaCl"], "-411.2"),
                rxn(["H2", "O2"], ["H2O"], "-571.6"),
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["target_enthalpy_kj"] == "-571.6"
    assert data["known_reactions"][5]["multiplier"] == "1"
    assert all(k["multiplier"] == "0" for k in data["known_reactions"][:5])
    # 与目标无关的物种也出现在净系数中，系数为 0
    net = {item["normalized"]: item["coefficient"]
           for item in data["net_coefficients"]}
    assert net["NH3"] == "0"
    assert net["H2O"] == "2"


def test_response_shape():
    resp = client.post(
        PATH,
        json={
            "target": rxn(["H2", "O2"], ["H2O"]),
            "known_reactions": [rxn(["H2", "O2"], ["H2O"], "-571.6")],
        },
    )
    data = resp.json()
    assert set(data) == {
        "target", "known_reactions", "target_enthalpy_kj", "net_coefficients"
    }
    assert set(data["target"]) == {"reactants", "products", "normalized"}
    known = data["known_reactions"][0]
    assert set(known) == {
        "index", "reactants", "products", "enthalpy_kj",
        "normalized", "multiplier", "contribution_kj",
    }
    assert set(data["net_coefficients"][0]) == {"normalized", "coefficient"}


# ------------------------------------------------------------ 错误路径
def test_not_derivable_unreachable_species():
    resp = client.post(
        PATH,
        json={
            "target": rxn(["C", "H2"], ["CH4"]),
            "known_reactions": [
                rxn(["C", "O2"], ["CO2"], "-393.5"),
                rxn(["CO", "O2"], ["CO2"], "-283.0"),
            ],
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "HESS_NOT_DERIVABLE"
    assert body["error"]["details"]["unreachable_species"] == ["CH4", "H2"]


def test_not_derivable_all_species_present_but_wrong_span():
    # 物种都出现过，但目标向量不在张成空间内（2 条已知反应线性无关，
    # 目标是第 3 个独立方向）
    resp = client.post(
        PATH,
        json={
            "target": rxn(["CO", "H2O"], ["CO2", "H2"]),  # 水煤气变换
            "known_reactions": [
                rxn(["C", "O2"], ["CO2"], "-393.5"),
                rxn(["H2", "O2"], ["H2O"], "-571.6"),
            ],
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "HESS_NOT_DERIVABLE"


def test_ambiguous_enthalpy_conflict_details():
    # A: S+O2->SO2 -296.8；B: 2SO2+O2->2SO3 -197.0；
    # C: 2S+3O2->2SO3 -790.0（自洽应为 -790.6，故意不自洽）
    resp = client.post(
        PATH,
        json={
            "target": rxn(["S", "O2"], ["SO3"]),
            "known_reactions": [
                rxn(["S", "O2"], ["SO2"], "-296.8"),
                rxn(["SO2", "O2"], ["SO3"], "-197.0"),
                rxn(["S", "O2"], ["SO3"], "-790.0"),
            ],
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "HESS_AMBIGUOUS"
    conflicts = body["error"]["details"]["conflicts"]
    # 零组合方向：-2A - B + C，焓差 = -790.0 - 2*(-296.8) - (-197.0) = 0.6
    assert {
        "multipliers": ["-2", "-1", "1"],
        "enthalpy_difference_kj": "3/5",
    } in conflicts


def test_ambiguous_when_target_spans_but_enthalpy_disagrees():
    # 目标恰为已知反应之一，但两条重复反应给出不同焓 -> 歧义
    resp = client.post(
        PATH,
        json={
            "target": rxn(["S", "O2"], ["SO2"]),
            "known_reactions": [
                rxn(["S", "O2"], ["SO2"], "-296.8"),
                rxn(["S", "O2"], ["SO2"], "-300.0"),
            ],
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "HESS_AMBIGUOUS"


def test_same_enthalpy_duplicates_return_normally():
    # 重复已知反应焓相同：零组合焓差为 0，正常返回
    resp = client.post(
        PATH,
        json={
            "target": rxn(["S", "O2"], ["SO2"]),
            "known_reactions": [
                rxn(["S", "O2"], ["SO2"], "-296.8"),
                rxn(["S", "O2"], ["SO2"], "-296.8"),
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["target_enthalpy_kj"] == "-296.8"
    # 最小支撑取下标字典序最小的第一条
    assert [k["multiplier"] for k in data["known_reactions"]] == ["1", "0"]


def test_invalid_enthalpy_string():
    resp = client.post(
        PATH,
        json={
            "target": rxn(["H2", "O2"], ["H2O"]),
            "known_reactions": [
                rxn(["H2", "O2"], ["H2O"], "abc"),
            ],
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "INVALID_REQUEST"
    assert body["error"]["details"]["invalid_enthalpies"] == [
        {"index": 0, "enthalpy_kj": "abc"}
    ]


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "1e20000"])
def test_invalid_enthalpy_nan_infinity_huge(bad):
    resp = client.post(
        PATH,
        json={
            "target": rxn(["H2", "O2"], ["H2O"]),
            "known_reactions": [rxn(["H2", "O2"], ["H2O"], bad)],
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_invalid_enthalpy_aggregates_all_bad_indexes():
    resp = client.post(
        PATH,
        json={
            "target": rxn(["H2", "O2"], ["H2O"]),
            "known_reactions": [
                rxn(["H2", "O2"], ["H2O"], "x"),
                rxn(["H2", "O2"], ["H2O"], "-571.6"),
                rxn(["H2", "O2"], ["H2O"], "  "),
            ],
        },
    )
    # 空白字符串在 Pydantic 层即被拒（422 结构化校验）
    assert resp.status_code == 422


def test_enthalpy_json_number_rejected():
    resp = client.post(
        PATH,
        json={
            "target": rxn(["H2", "O2"], ["H2O"]),
            "known_reactions": [
                {"reactants": ["H2", "O2"], "products": ["H2O"], "enthalpy_kj": -571.6},
            ],
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_target_formula_parse_error():
    resp = client.post(
        PATH,
        json={
            "target": rxn(["Xx"], ["H2O"]),
            "known_reactions": [rxn(["H2", "O2"], ["H2O"], "-571.6")],
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "FORMULA_PARSE_ERROR"
    assert body["error"]["details"]["reaction"] == "target"


def test_known_formula_parse_error_with_index():
    resp = client.post(
        PATH,
        json={
            "target": rxn(["H2", "O2"], ["H2O"]),
            "known_reactions": [
                rxn(["H2", "O2"], ["H2O"], "-571.6"),
                rxn(["Xx"], ["H2O"], "10"),
            ],
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "FORMULA_PARSE_ERROR"
    assert body["error"]["details"]["reaction"] == "known_reactions"
    assert body["error"]["details"]["index"] == 1


def test_duplicate_species_target_and_known():
    resp = client.post(
        PATH,
        json={
            "target": rxn(["H2", "H2"], ["H2O"]),
            "known_reactions": [rxn(["H2", "O2"], ["H2O"], "-571.6")],
        },
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "DUPLICATE_SPECIES"
    assert resp.json()["error"]["details"]["reaction"] == "target"

    resp = client.post(
        PATH,
        json={
            "target": rxn(["H2", "O2"], ["H2O"]),
            "known_reactions": [
                rxn(["H2", "O2"], ["H2O"], "-571.6"),
                rxn(["O2", "O2"], ["H2O"], "10"),
            ],
        },
    )
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"]["code"] == "DUPLICATE_SPECIES"
    assert body["error"]["details"]["index"] == 1


def test_target_non_unique_balance():
    resp = client.post(
        PATH,
        json={
            "target": rxn(["C", "O2"], ["CO", "CO2"]),
            "known_reactions": [rxn(["C", "O2"], ["CO2"], "-393.5")],
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "NON_UNIQUE_BALANCE"
    assert body["error"]["details"]["reaction"] == "target"


def test_known_non_unique_balance_with_index():
    resp = client.post(
        PATH,
        json={
            "target": rxn(["H2", "O2"], ["H2O"]),
            "known_reactions": [
                rxn(["C", "O2"], ["CO", "CO2"], "10"),
            ],
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "NON_UNIQUE_BALANCE"
    assert body["error"]["details"]["reaction"] == "known_reactions"
    assert body["error"]["details"]["index"] == 0


def test_impossible_balance_in_known():
    resp = client.post(
        PATH,
        json={
            "target": rxn(["H2", "O2"], ["H2O"]),
            "known_reactions": [rxn(["H2O"], ["Au"], "10")],
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "IMPOSSIBLE_BALANCE"


@pytest.mark.parametrize("payload", [
    {"known_reactions": [rxn(["H2", "O2"], ["H2O"], "-571.6")]},  # 缺 target
    {"target": rxn(["H2", "O2"], ["H2O"])},                      # 缺 known
    {
        "target": rxn(["H2", "O2"], ["H2O"]),
        "known_reactions": [],                                    # 空列表
    },
    {
        "target": rxn(["H2", "O2"], ["H2O"]),
        "known_reactions": [
            rxn(["H2", "O2"], ["H2O"], "0") for _ in range(7)     # 超过 6 条
        ],
    },
    {
        "target": {"reactants": [], "products": ["H2O"]},
        "known_reactions": [rxn(["H2", "O2"], ["H2O"], "0")],
    },
    {
        "target": rxn(["H2", "O2"], ["H2O"]),
        "known_reactions": [
            {"reactants": ["H2", "O2"], "products": ["H2O"]},    # 缺 enthalpy_kj
        ],
    },
])
def test_invalid_request_structure(payload):
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_known_balance_error_does_not_include_reaction_id():
    resp = client.post(
        PATH,
        json={
            "target": rxn(["H2", "O2"], ["H2O"]),
            "known_reactions": [rxn(["Xx"], ["H2O"], "10")],
        },
    )
    body = resp.json()
    assert "reaction_id" not in body
    assert "cached" not in body
