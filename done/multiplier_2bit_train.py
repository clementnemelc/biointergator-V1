"""
multiplier_2bit_train_clean.py
2-bit SNN Multiplier 穩定性測試 (精簡版)
- AND gate 固定正確，不訓練
- 只訓練 Half-Adder
- 關閉 homeostasis 影響 (HS_SCALE=1.1，處理 +/- 0.2 權重波動)
- 測試每個 seed：先檢查 HA 4/4 是否正確，再檢查乘法器 16/16
"""

import os
import random
import sys

import numpy as np

from functions import *


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# ============================================================
# 超參數
# ============================================================
THRESHOLD = 0.6
TICKS = 12
N_EPOCHS = 400
LTP = 0.04
LTD = 0.15
N_SILENT = 3
HS_SCALE = 1.2  # 提高 HS_SCALE 以應對 0.2 波動


def reset_state(net):
    for layer in net:
        for n in layer:
            n["state"].update({"v": 0.0, "spike": 0.0})
            n["input_buffer"] = [0.0] * len(n["input_buffer"])
            n["refrac_abs"] = 0
            n["refrac_rel"] = 0


def run_trial(net, inputs, ticks=TICKS):
    reset_state(net)
    inject_input(net[0], inputs)
    fired_flags = [False] * len(net[-1])
    fired_set = set()
    for _ in range(ticks):
        step(net)
        for layer in net:
            for n in layer:
                if n["state"]["spike"] > 0:
                    fired_set.add(id(n))
        for i, out_n in enumerate(net[-1]):
            if out_n["state"]["spike"] > 0:
                fired_flags[i] = True
    return [int(f) for f in fired_flags], fired_set


# ============================================================
# 模組：Half-Adder 訓練
# ============================================================
def create_half_adder_init():
    clear_synapses()
    in_l = create_layer(
        2, name="hin", threshold_range=(THRESHOLD, THRESHOLD), excitatory=True
    )
    hid_l = create_layer(
        1, name="hhid", threshold_range=(THRESHOLD, THRESHOLD), excitatory=False
    )
    out_l = create_layer(
        2, name="hout", threshold_range=(THRESHOLD, THRESHOLD), excitatory=True
    )

    # 初始權重：根據 table 基準值 +/- 0.2
    # In -> Hid: 0.4 +/- 0.2 -> [0.2, 0.6]
    hid_w = random.uniform(0.2, 0.6)
    fully_connect(in_l, hid_l, weight_range=(hid_w, hid_w))

    # In -> Sum: 1.0 +/- 0.2 -> [0.8, 1.2]
    base_sum = random.uniform(0.8, 1.2)
    connect_modified(
        in_l, out_l, from_idx=0, to_idx=0, weight_range=(base_sum, base_sum), delay=1
    )
    connect_modified(
        in_l, out_l, from_idx=1, to_idx=0, weight_range=(base_sum, base_sum), delay=1
    )

    # In -> Carry: 0.4 +/- 0.2 -> [0.2, 0.6]
    base_carry = random.uniform(0.2, 0.6)
    connect_modified(
        in_l,
        out_l,
        from_idx=0,
        to_idx=1,
        weight_range=(base_carry, base_carry),
        delay=1,
    )
    connect_modified(
        in_l,
        out_l,
        from_idx=1,
        to_idx=1,
        weight_range=(base_carry, base_carry),
        delay=1,
    )

    # Hid -> Sum: 2.5 +/- 0.2 -> [2.3, 2.7]
    inh_w = random.uniform(2.3, 2.7)
    fully_connect(hid_l, out_l, weight_range=(inh_w, inh_w))

    return [in_l, hid_l, out_l]


def train_half_adder():
    net = create_half_adder_init()
    stdp = STDPRule(ltp=LTP, ltd=LTD, w_min=0.1, w_max=1.0)

    truth = [(0, 0), (0, 1), (1, 0), (1, 1)]
    expected = [[0, 0], [1, 0], [1, 0], [0, 1]]

    for epoch in range(N_EPOCHS):
        correct = 0
        all_epoch_fired = set()
        for inp, exp in zip(truth, expected):
            out, f_set = run_trial(net, inp)
            all_epoch_fired |= f_set
            if out == exp:
                correct += 1

            # 寫回輸出 spike 狀態
            for i, n in enumerate(net[-1]):
                n["state"]["spike"] = float(out[i])

            # 只在錯誤樣本上做 STDP (輸出層)
            if out != exp:
                with HiddenPrints():
                    stdp.apply_all(
                        net[-1],
                        [o == e for o, e in zip(out, exp)],
                        fired_set=f_set,
                    )

            # HA 隱藏層訓練 (針對 XOR 邏輯)
            hid_actual = [1 if n["state"]["spike"] > 0 else 0 for n in net[1]]
            hid_expected = [1 if (inp[0] and inp[1]) else 0]
            for i, hid_n in enumerate(net[1]):
                hid_n["state"]["spike"] = float(hid_actual[i])
            if hid_actual != hid_expected:
                with HiddenPrints():
                    stdp.apply_all(
                        net[1],
                        [a == e for a, e in zip(hid_actual, hid_expected)],
                        fired_set=f_set,
                    )

        # 體內平衡
        if HS_SCALE > 1.0:
            with HiddenPrints():
                stdp.homeostatic_scale(
                    net[-1], all_epoch_fired, n_silent=N_SILENT, scale=HS_SCALE
                )
                stdp.homeostatic_scale(
                    net[1], all_epoch_fired, n_silent=N_SILENT, scale=HS_SCALE
                )

        if correct == len(truth):
            return net, epoch + 1

    return net, -1  # 未收斂標記


def check_half_adder(net):
    truth = [(0, 0), (0, 1), (1, 0), (1, 1)]
    expected = [[0, 0], [1, 0], [1, 0], [0, 1]]
    for inp, exp in zip(truth, expected):
        out, _ = run_trial(net, inp)
        if out != exp:
            return False
    return True


# ============================================================
# 模組：AND gate 訓練
# ============================================================
def create_and_gate_init():
    clear_synapses()
    in_l = create_layer(
        2, name="ain", threshold_range=(THRESHOLD, THRESHOLD), excitatory=True
    )
    out_l = create_layer(
        1, name="aout", threshold_range=(THRESHOLD, THRESHOLD), excitatory=True
    )
    # 0.4 +/- 0.2 -> [0.2, 0.6]
    w = random.uniform(0.2, 0.6)
    connect_modified(in_l, out_l, from_idx=0, to_idx=0, weight_range=(w, w), delay=1)
    connect_modified(in_l, out_l, from_idx=1, to_idx=0, weight_range=(w, w), delay=1)
    return [in_l, out_l]


def train_and_gate():
    net = create_and_gate_init()
    stdp = STDPRule(ltp=LTP, ltd=LTD, w_min=0.1, w_max=1.0)
    truth = [(0, 0), (0, 1), (1, 0), (1, 1)]
    expected = [[0], [0], [0], [1]]

    for epoch in range(N_EPOCHS):
        correct = 0
        all_epoch_fired = set()
        for inp, exp in zip(truth, expected):
            out, f_set = run_trial(net, inp)
            all_epoch_fired |= f_set
            if out == exp:
                correct += 1
            for i, n in enumerate(net[-1]):
                n["state"]["spike"] = float(out[i])
            if out != exp:
                with HiddenPrints():
                    stdp.apply_all(
                        net[-1], [o == e for o, e in zip(out, exp)], fired_set=f_set
                    )

        if HS_SCALE > 1.0:
            with HiddenPrints():
                stdp.homeostatic_scale(
                    net[-1], all_epoch_fired, n_silent=N_SILENT, scale=HS_SCALE
                )

        if correct == len(truth):
            return net, epoch + 1
    return net, -1


# ============================================================
# 2-bit 乘法器：組合 AND + HA
# ============================================================
def build_and_nets():
    nets = []
    for i in range(4):
        net, ep = train_and_gate()
        if ep == -1:
            return None
        nets.append(net)
    return nets


def run_and_gate_net(net, A, B):
    out, _ = run_trial(net, (A, B))
    return out[0]


def run_half_adder_net(net, A, B):
    out, _ = run_trial(net, (A, B))
    return out[0], out[1]  # Sum, Carry


def run_multiplier(and_nets, ha_nets, A, B):
    a0, a1 = (A & 1), (A >> 1) & 1
    b0, b1 = (B & 1), (B >> 1) & 1

    # 部分積
    pp0 = run_and_gate_net(and_nets[0], a0, b0)
    pp1 = run_and_gate_net(and_nets[1], a1, b0)
    pp2 = run_and_gate_net(and_nets[2], a0, b1)
    pp3 = run_and_gate_net(and_nets[3], a1, b1)

    # 組合
    p1, c1 = run_half_adder_net(ha_nets[0], pp1, pp2)
    p2, p3 = run_half_adder_net(ha_nets[1], pp3, c1)

    return (p3 << 3) | (p2 << 2) | (p1 << 1) | pp0


# ============================================================
# Main：100 seeds 測試
# ============================================================
seeds = 100
multiplier_success = 0

print(f"=== 2-Bit Multiplier Clean Stability Test ({seeds} Seeds) ===")

for s in range(seeds):
    random.seed(s)
    np.random.seed(s)

    # AND 模組固定正確
    and_nets = build_and_nets()
    if and_nets is None:
        continue

    # 訓練兩個 Half-Adder
    ha_nets = []
    ok_all = True
    for i in range(2):
        net, ep = train_half_adder()
        if ep == -1 or not check_half_adder(net):
            ok_all = False
            break
        ha_nets.append(net)

    if not ok_all:
        continue

    # 測整體乘法器
    all_correct = True
    for A in range(4):
        for B in range(4):
            res = run_multiplier(and_nets, ha_nets, A, B)
            if res != A * B:
                all_correct = False
                break
        if not all_correct:
            break

    if all_correct:
        multiplier_success += 1

    if (s + 1) % 10 == 0:
        print(
            f" Progress: {s + 1}/{seeds} seeds, "
            f"success so far: {multiplier_success / (s + 1) * 100:.1f}%"
        )

print("\n" + "=" * 50)
print(f"Final Multiplier Success Rate: {multiplier_success / seeds * 100:.1f}%")
print("=" * 50)
