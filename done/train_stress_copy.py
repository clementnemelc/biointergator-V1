"""
train_stress_copy.py
壓力測試：隨機設定權重執行 100 個 seeds，統計收斂至 acc=1.00 的 EPOCHS
"""

import os
import random
import sys

from functions import *


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# ============================================================
# 1. 超參數
# ============================================================
THRESHOLD = 0.5
TICKS = 8
N_EPOCHS = 300
LTP = 0.04
LTD = 0.15
N_SILENT = 3
HS_SCALE = 1.5

TRUTH_TABLE = [(0, 0), (0, 1), (1, 0), (1, 1)]
EXPECTED = [(0, 0), (1, 0), (1, 0), (0, 1)]


# ============================================================
# 2. 定義輔助函數
# ============================================================
def reset_state(net):
    for layer in net:
        for n in layer:
            n["state"].update({"v": 0.0, "spike": 0.0})
            n["input_buffer"] = [0.0] * len(n["input_buffer"])
            n["refrac_abs"] = 0
            n["refrac_rel"] = 0


def run_trial(net, A, B, ticks):
    reset_state(net)
    inject_input(net[0], [A, B])
    sf = cf = False
    fired_set = set()
    for _ in range(ticks):
        step(net)
        for layer in net:
            for n in layer:
                if n["state"]["spike"] > 0:
                    fired_set.add(id(n))
        if net[2][0]["state"]["spike"] > 0:
            sf = True
        if net[2][1]["state"]["spike"] > 0:
            cf = True
    return sf, cf, fired_set


# ============================================================
# Grid Search for LTP / LTD Factor
# ============================================================
import itertools

import numpy as np

print("\n=== Starting Grid Search for STDP Parameters (THRESHOLD=0.6) ===")
LTP_CANDIDATES = [0.03, 0.04, 0.05, 0.06]
FACTOR_CANDIDATES = [4.0, 5.0, 6.0, 7.0, 8.0]

best_rate = -1
best_params = None
results = []

for ltp_base, factor in itertools.product(LTP_CANDIDATES, FACTOR_CANDIDATES):
    ltd_base = round(ltp_base * factor, 4)
    print(f"\n--- Testing LTP: {ltp_base}, LTD: {ltd_base} (Factor: {factor}) ---")

    epochs_to_converge = []

    for SEED in range(100):
        random.seed(SEED)

        clear_synapses()
        T = 0.6
        create_layer(2, name="in_put", threshold_range=(T, T), excitatory=True)
        create_layer(1, name="hid", threshold_range=(T, T), excitatory=False)
        create_layer(2, name="out_put", threshold_range=(T, T), excitatory=True)
        network = [in_put, hid, out_put]

        # 隨機初始化權重範圍
        # in_put -> hid (延遲 delay=1)
        fully_connect(in_put, hid, weight_range=(0.2, 0.6))

        # in_put -> Sum (out_put_0): 增加延遲到 delay=2，等待抑制訊號同步抵達
        connect_modified(
            in_put, out_put, from_idx=0, to_idx=0, weight_range=(0.8, 1.2), delay=2
        )
        connect_modified(
            in_put, out_put, from_idx=1, to_idx=0, weight_range=(0.8, 1.2), delay=2
        )

        # in_put -> Carry (out_put_1): 延遲維持 1
        connect_modified(
            in_put, out_put, from_idx=0, to_idx=1, weight_range=(0.2, 0.6), delay=1
        )
        connect_modified(
            in_put, out_put, from_idx=1, to_idx=1, weight_range=(0.2, 0.6), delay=1
        )

        # hid -> out_put (延遲 1)
        fully_connect(hid, out_put, weight_range=(1.8, 2.2))

        # Base STDP parameter as chosen from previous search
        stdp = STDPRule(ltp=ltp_base, ltd=ltd_base, w_min=0.1, w_max=1.0)
        learning_locked = False
        consecutive_failures = 0

        converged_epoch = -1
        for epoch in range(N_EPOCHS):
            n_correct = 0
            bit_errors = 0
            epoch_fired = set()

            for (A, B), (exp_s, exp_c) in zip(TRUTH_TABLE, EXPECTED):
                sf, cf, fired_set = run_trial(network, A, B, ticks=8)
                epoch_fired |= fired_set

                sum_ok = int(sf) == exp_s
                carry_ok = int(cf) == exp_c
                if sum_ok and carry_ok:
                    n_correct += 1
                else:
                    if learning_locked:
                        learning_locked = False  # 發生錯誤時解鎖

                bit_errors += (0 if sum_ok else 1) + (0 if carry_ok else 1)

                out_put[0]["state"]["spike"] = float(sf)
                out_put[1]["state"]["spike"] = float(cf)

                with HiddenPrints():
                    if not learning_locked:
                        stdp.apply_all(out_put, [sum_ok, carry_ok], fired_set=fired_set)

            acc = n_correct / len(TRUTH_TABLE)

            if acc < 1.0:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    # 動態調整：連續 30 次未正確時，減少 LTP，增加 LTD
                    stdp.ltp = round(max(stdp.ltp * 0.09, 0.001), 4)
                    stdp.ltd = round(min(stdp.ltd * 1.01, 0.5), 4)
                    consecutive_failures = 0  # 調整後歸零重新計算
            else:
                consecutive_failures = 0

            with HiddenPrints():
                if not learning_locked:
                    stdp.homeostatic_scale(out_put, epoch_fired, n_silent=3, scale=1.5)

            if acc == 1.0:
                converged_epoch = epoch + 1
                break

        if converged_epoch == -1:
            converged_epoch = N_EPOCHS + 1  # means not converged or > N_EPOCHS

        epochs_to_converge.append(converged_epoch)

        # 簡易進度條
        if (SEED + 1) % 50 == 0:
            print(f"  Processed {SEED + 1} / 100 seeds...", end="\r")
    print()  # New line after loading 100 seeds

    # 統計
    successful_epochs = [ep for ep in epochs_to_converge if ep <= N_EPOCHS]
    success_rate = len(successful_epochs) / 100.0
    avg_epoch = np.mean(successful_epochs) if successful_epochs else float("inf")

    print(
        f"  Result: Success Rate = {success_rate * 100:.1f}%, Avg Epochs = {avg_epoch:.1f}"
    )
    results.append((ltp_base, factor, success_rate, avg_epoch))

    if success_rate > best_rate or (
        success_rate == best_rate and avg_epoch < best_params[3]
    ):
        best_rate = success_rate
        best_params = (ltp_base, factor, success_rate, avg_epoch)

# ============================================================
# 總結
# ============================================================
print("\n" + "=" * 50)
print("Grid Search Results (Sorted by Success Rate, then Avg Epochs):")
# Sort by descending success rate, then ascending epochs
results.sort(key=lambda x: (-x[2], x[3]))
for ltp, factor, rate, avg_ep in results:
    s_ep = f"{avg_ep:.1f}" if avg_ep != float("inf") else "N/A"
    print(
        f"LTP: {ltp:0.3f} | Factor: {factor:0.1f} | LTD: {ltp * factor:0.3f} | Rate: {rate * 100:5.1f}% | Avg Epoch: {s_ep}"
    )

print("=" * 50)
if best_params:
    print(
        f"Best Configuration: LTP = {best_params[0]}, LTD = {best_params[0] * best_params[1]:.4f} (Factor {best_params[1]})"
    )
    print(
        f"Achieved Rate: {best_params[2] * 100}% in {best_params[3]:.1f} epochs average."
    )
