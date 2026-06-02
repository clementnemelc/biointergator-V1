"""
search_params_multiplier.py
使用網格搜索尋找適合 +/- 0.2 權重波動下的 LTP/LTD 參數。
使用 20 個 Seed 進行快速評估。
"""

import os
import random
import sys

import numpy as np

from functions import *


# 隱藏打印
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# ============================================================
# 基本配置 (由 multiplier_2bit_train.py 移植)
# ============================================================
THRESHOLD = 0.5
TICKS = 12
N_EPOCHS = 400


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
    hid_w = random.uniform(0.2, 0.6)
    fully_connect(in_l, hid_l, weight_range=(hid_w, hid_w))
    base_sum = random.uniform(0.8, 1.2)
    connect_modified(
        in_l, out_l, from_idx=0, to_idx=0, weight_range=(base_sum, base_sum), delay=1
    )
    connect_modified(
        in_l, out_l, from_idx=1, to_idx=0, weight_range=(base_sum, base_sum), delay=1
    )
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
    inh_w = random.uniform(2.3, 2.7)
    fully_connect(hid_l, out_l, weight_range=(inh_w, inh_w))
    return [in_l, hid_l, out_l]


def create_and_gate_init():
    clear_synapses()
    in_l = create_layer(
        2, name="ain", threshold_range=(THRESHOLD, THRESHOLD), excitatory=True
    )
    out_l = create_layer(
        1, name="aout", threshold_range=(THRESHOLD, THRESHOLD), excitatory=True
    )
    w = random.uniform(0.2, 0.6)
    connect_modified(in_l, out_l, from_idx=0, to_idx=0, weight_range=(w, w), delay=1)
    connect_modified(in_l, out_l, from_idx=1, to_idx=0, weight_range=(w, w), delay=1)
    return [in_l, out_l]


def train_module(create_func, truth, expected, ltp, ltd):
    net = create_func()
    stdp = STDPRule(ltp=ltp, ltd=ltd, w_min=0.1, w_max=1.0)
    for epoch in range(N_EPOCHS):
        correct = 0
        for inp, exp in zip(truth, expected):
            out, f_set = run_trial(net, inp)
            if out == exp:
                correct += 1
            for i, n in enumerate(net[-1]):
                n["state"]["spike"] = float(out[i])
            if out != exp:
                with HiddenPrints():
                    stdp.apply_all(
                        net[-1], [o == e for o, e in zip(out, exp)], fired_set=f_set
                    )
        if correct == len(truth):
            return net, epoch + 1
    return net, -1


def run_multiplier(and_nets, ha_nets, A, B):
    a0, a1 = (A & 1), (A >> 1) & 1
    b0, b1 = (B & 1), (B >> 1) & 1
    pp = [
        run_trial(n, (a0 if i in [0, 2] else a1, b0 if i in [0, 1] else b1))[0][0]
        for i, n in enumerate(and_nets)
    ]
    p1, c1 = run_trial(ha_nets[0], (pp[1], pp[2]))[0]
    p2, p3 = run_trial(ha_nets[1], (pp[3], c1))[0]
    return (p3 << 3) | (p2 << 2) | (p1 << 1) | pp[0]


def evaluate(ltp, ltd, num_seeds=20):
    success = 0
    for s in range(num_seeds):
        random.seed(s)
        np.random.seed(s)

        and_nets = []
        for _ in range(4):
            net, ep = train_module(
                create_and_gate_init,
                [(0, 0), (0, 1), (1, 0), (1, 1)],
                [[0], [0], [0], [1]],
                ltp,
                ltd,
            )
            if ep == -1:
                break
            and_nets.append(net)
        if len(and_nets) < 4:
            continue

        ha_nets = []
        for _ in range(2):
            net, ep = train_module(
                create_half_adder_init,
                [(0, 0), (0, 1), (1, 0), (1, 1)],
                [[0, 0], [1, 0], [1, 0], [0, 1]],
                ltp,
                ltd,
            )
            if ep == -1:
                break
            ha_nets.append(net)
        if len(ha_nets) < 2:
            continue

        all_correct = True
        for A in range(4):
            for B in range(4):
                if run_multiplier(and_nets, ha_nets, A, B) != A * B:
                    all_correct = False
                    break
            if not all_correct:
                break
        if all_correct:
            success += 1
    return success / num_seeds


# ============================================================
# Grid Search
# ============================================================
LTP_LIST = [0.06, 0.08, 0.10]
LTD_LIST = [0.20, 0.25, 0.30]

print(f"{'LTP':>6} | {'LTD':>6} | {'Success%':>8}", flush=True)
print("-" * 28, flush=True)

best_acc = -1
best_params = (0, 0)

for ltp in LTP_LIST:
    for ltd in LTD_LIST:
        acc = evaluate(ltp, ltd, num_seeds=10)  # 減少 seed 數以便觀察
        print(f"{ltp:6.3f} | {ltd:6.3f} | {acc * 100:7.1f}%", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_params = (ltp, ltd)

print("-" * 28, flush=True)
print(
    f"Best: LTP={best_params[0]}, LTD={best_params[1]} (Acc: {best_acc * 100:.1f}%)",
    flush=True,
)
