"""
multiplier_2bit.py
2-bit SNN Multiplier
A (A1A0) × B (B1B0) = P (P3P2P1P0)

Logic:
1. 4 Partial Products (AND gates):
   pp0 = A0 & B0
   pp1 = A1 & B0
   pp2 = A0 & B1
   pp3 = A1 & B1

2. Summing:
   P0 = pp0
   P1, C1 = HalfAdder(pp1, pp2)
   P2, P3 = HalfAdder(pp3, C1)
"""

from functions import *

# ============================================================
# 超參數
# ============================================================
T = 0.5
TICKS = 8


# ============================================================
# 輔助組件函數
# ============================================================
def reset_state(net):
    for layer in net:
        for n in layer:
            n["state"].update({"v": 0.0, "spike": 0.0})
            n["input_buffer"] = [0.0] * len(n["input_buffer"])
            n["refrac_abs"] = 0


def run_and_gate(A, B):
    """1-bit AND Gate"""
    clear_synapses()
    # 建立層
    in_layer = create_layer(2, name="and_in", threshold_range=(T, T), excitatory=True)
    out_layer = create_layer(1, name="and_out", threshold_range=(T, T), excitatory=True)

    # 布線：AND 邏輯 (0.4 + 0.4 = 0.8 > 0.5)
    connect_modified(
        in_layer, out_layer, from_idx=0, to_idx=0, weight_range=(0.4, 0.4), delay=0
    )
    connect_modified(
        in_layer, out_layer, from_idx=1, to_idx=0, weight_range=(0.4, 0.4), delay=0
    )

    net = [in_layer, out_layer]
    reset_state(net)
    inject_input(in_layer, [A, B])

    fired = False
    for _ in range(TICKS):
        step(net)
        if out_layer[0]["state"]["spike"] > 0:
            fired = True
    return 1 if fired else 0


def run_half_adder(A, B):
    """Half Adder (Sum, Carry)"""
    clear_synapses()
    in_l = create_layer(2, name="ha_in", threshold_range=(T, T), excitatory=True)
    hid_l = create_layer(1, name="ha_hid", threshold_range=(T, T), excitatory=False)
    out_l = create_layer(2, name="ha_out", threshold_range=(T, T), excitatory=True)

    # Sum (OR part)
    connect_modified(
        in_l, out_l, from_idx=0, to_idx=0, weight_range=(1.0, 1.0), delay=2
    )
    connect_modified(
        in_l, out_l, from_idx=1, to_idx=0, weight_range=(1.0, 1.0), delay=2
    )

    # Carry (AND part)
    connect_modified(
        in_l, out_l, from_idx=0, to_idx=1, weight_range=(0.4, 0.4), delay=2
    )
    connect_modified(
        in_l, out_l, from_idx=1, to_idx=1, weight_range=(0.4, 0.4), delay=2
    )

    # XOR Inhibition (in -> hid -> sum)
    connect_modified(
        in_l, hid_l, from_idx=0, to_idx=0, weight_range=(0.4, 0.4), delay=1
    )
    connect_modified(
        in_l, hid_l, from_idx=1, to_idx=0, weight_range=(0.4, 0.4), delay=1
    )
    connect_modified(
        hid_l, out_l, from_idx=0, to_idx=0, weight_range=(2.0, 2.0), delay=1
    )

    net = [in_l, hid_l, out_l]
    reset_state(net)
    inject_input(in_l, [A, B])

    sum_fired = False
    carry_fired = False
    for _ in range(TICKS):
        step(net)
        if out_l[0]["state"]["spike"] > 0:
            sum_fired = True
        if out_l[1]["state"]["spike"] > 0:
            carry_fired = True
    return (1 if sum_fired else 0), (1 if carry_fired else 0)


# ============================================================
# 2-Bit 乘法主邏輯
# ============================================================
def multiplier_2bit(A, B):
    # 分解 A1A0, B1B0
    a0, a1 = (A & 1), (A >> 1) & 1
    b0, b1 = (B & 1), (B >> 1) & 1

    # 1. 偏積 (Partial Products)
    pp0 = run_and_gate(a0, b0)
    pp1 = run_and_gate(a1, b0)
    pp2 = run_and_gate(a0, b1)
    pp3 = run_and_gate(a1, b1)

    # 2. 組合邏輯
    p0 = pp0
    p1, c1 = run_half_adder(pp1, pp2)
    p2, p3 = run_half_adder(pp3, c1)

    return p0, p1, p2, p3


# ============================================================
# 測試
# ============================================================
print("=== 2-Bit SNN Multiplier Test (0-3 x 0-3) ===")
print("A  x  B  |  P3 P2 P1 P0  (Dec) | Expected | Match")
print("-" * 50)

for A in range(4):
    for B in range(4):
        p0, p1, p2, p3 = multiplier_2bit(A, B)
        res_dec = (p3 << 3) | (p2 << 2) | (p1 << 1) | p0
        expected = A * B
        match = "OK" if res_dec == expected else "FAIL"
        print(
            f"{A}  x  {B}  |  {p3}  {p2}  {p1}  {p0}   ({res_dec:2}) |    {expected:2}    | {match}"
        )
