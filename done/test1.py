"""
train_stress_copy.py
壓力測試：隨機設定權重執行 100 個 seeds，統計收斂至 acc=1.00 的 EPOCHS
改版重點：
1. 固定 delay 結構不變。
2. in->hid 下界提高，確保 AND 抑制足夠強。
3. Sum/Carry 成對連結採「base ± 小擾動」對稱初始化。
4. 移除過度激進的 LTP/LTD 鎖死機制，先用固定 STDP 觀察收斂率。
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
# 1. 超參數
# ============================================================
THRESHOLD = 0.5
TICKS = 8
N_EPOCHS = 300

# 推薦落在你前面 grid search 的穩定區域
BASE_LTP = 0.04
BASE_LTD = 0.24  # 6 倍 LTP

N_SILENT = 3
HS_SCALE = 2.0  # 中間值，實驗顯示 1.5~3.0 差不多

TRUTH_TABLE = [(0, 0), (0, 1), (1, 0), (1, 1)]
EXPECTED = [(0, 0), (1, 0), (1, 0), (0, 1)]


# ============================================================
# 2. 輔助函數
# ============================================================
def reset_state(net):
    for layer in net:
        for n in layer:
            n["state"].update({"v": 0.0, "spike": 0.0})
            n["input_buffer"] = [0.0] * len(n["input_buffer"])
            n["refrac_abs"] = 0
            n["refrac_rel"] = 0


def run_trial(net, A, B):
    reset_state(net)
    inject_input(net[0], [A, B])
    sf = cf = False
    fired_set = set()
    for _ in range(TICKS):
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
# 3. 主實驗：100 seeds，固定 STDP，對稱初始化
# ============================================================
print("\n=== Starting Seed Test with Symmetric Init & Fixed STDP ===")

epochs_to_converge = []
init_records = []  # 用來記錄每個 seed 的初始權重 + 是否收斂

for SEED in range(100):
    random.seed(SEED)

    clear_synapses()
    T = THRESHOLD
    create_layer(2, name="in_put", threshold_range=(T, T), excitatory=True)
    create_layer(1, name="hid", threshold_range=(T, T), excitatory=False)
    create_layer(2, name="out_put", threshold_range=(T, T), excitatory=True)
    network = [in_put, hid, out_put]

    # ---------- 3.1 in_put -> hid (AND 抑制路徑，延遲 1) ----------
    # 收窄範圍並抬高下界，避免抑制太弱
    hid_w = random.uniform(0.35, 0.5)
    fully_connect(in_put, hid, weight_range=(hid_w, hid_w))

    # ---------- 3.2 in_put -> Sum (out_put_0)，延遲 2 ----------
    # 用 base ± 小擾動，確保兩條連結大致對稱
    base_sum = random.uniform(0.9, 1.1)
    sum_eps = 0.05
    w_sum_0 = max(0.1, min(1.0, base_sum + random.uniform(-sum_eps, sum_eps)))
    w_sum_1 = max(0.1, min(1.0, base_sum + random.uniform(-sum_eps, sum_eps)))

    connect_modified(
        in_put, out_put, from_idx=0, to_idx=0, weight_range=(w_sum_0, w_sum_0), delay=2
    )
    connect_modified(
        in_put, out_put, from_idx=1, to_idx=0, weight_range=(w_sum_1, w_sum_1), delay=2
    )

    # ---------- 3.3 in_put -> Carry (out_put_1)，延遲 1 ----------
    base_carry = random.uniform(0.35, 0.55)
    carry_eps = 0.05
    w_carry_0 = max(0.1, min(1.0, base_carry + random.uniform(-carry_eps, carry_eps)))
    w_carry_1 = max(0.1, min(1.0, base_carry + random.uniform(-carry_eps, carry_eps)))

    connect_modified(
        in_put,
        out_put,
        from_idx=0,
        to_idx=1,
        weight_range=(w_carry_0, w_carry_0),
        delay=1,
    )
    connect_modified(
        in_put,
        out_put,
        from_idx=1,
        to_idx=1,
        weight_range=(w_carry_1, w_carry_1),
        delay=1,
    )

    # ---------- 3.4 hid -> out_put (抑制，全連接，延遲 1) ----------
    inh_w = random.uniform(1.9, 2.1)
    fully_connect(hid, out_put, weight_range=(inh_w, inh_w))

    # 紀錄這個 seed 的初始權重
    init_records.append(
        {
            "seed": SEED,
            "w_in0_hid": hid_w,
            "w_in1_hid": hid_w,
            "w_in0_sum": w_sum_0,
            "w_in1_sum": w_sum_1,
            "w_in0_carry": w_carry_0,
            "w_in1_carry": w_carry_1,
            "w_hid_sum": inh_w,
            "w_hid_carry": inh_w,
        }
    )

    # ---------- 建立 STDP ----------
    stdp = STDPRule(ltp=BASE_LTP, ltd=BASE_LTD, w_min=0.1, w_max=1.0)

    converged_epoch = -1

    for epoch in range(N_EPOCHS):
        n_correct = 0
        epoch_fired = set()

        for (A, B), (exp_s, exp_c) in zip(TRUTH_TABLE, EXPECTED):
            sf, cf, fired_set = run_trial(network, A, B)
            epoch_fired |= fired_set

            sum_ok = int(sf) == exp_s
            carry_ok = int(cf) == exp_c
            if sum_ok and carry_ok:
                n_correct += 1

            out_put[0]["state"]["spike"] = float(sf)
            out_put[1]["state"]["spike"] = float(cf)

            # 建議：只在有錯誤時更新 STDP（可視需要改成註解/啟用）
            if not (sum_ok and carry_ok):
                with HiddenPrints():
                    stdp.apply_all(out_put, [sum_ok, carry_ok], fired_set=fired_set)

        acc = n_correct / len(TRUTH_TABLE)

        with HiddenPrints():
            stdp.homeostatic_scale(
                out_put, epoch_fired, n_silent=N_SILENT, scale=HS_SCALE
            )

        if acc == 1.0:
            converged_epoch = epoch + 1
            break

    if converged_epoch == -1:
        converged_epoch = N_EPOCHS + 1  # 未收斂或超過限制

    epochs_to_converge.append(converged_epoch)
    init_records[-1]["converged"] = int(converged_epoch <= N_EPOCHS)
    init_records[-1]["epoch"] = converged_epoch

    # 簡易進度顯示
    if (SEED + 1) % 20 == 0:
        print(f" Processed {SEED + 1} / 100 seeds...")

# ============================================================
# 4. 統計與輸出
# ============================================================
successful_epochs = [ep for ep in epochs_to_converge if ep <= N_EPOCHS]
success_rate = len(successful_epochs) / 100.0
avg_epoch = np.mean(successful_epochs) if successful_epochs else float("inf")

print("\n=== Convergence Statistics (100 Seeds) ===")
print("Total Seeds: 100")
print(f"Success Rate: {success_rate * 100:.1f}%")
print(f"Avg Converged Epoch: {avg_epoch:.1f}")

group_less_than_2 = sum(1 for ep in successful_epochs if ep < 2)
group_2_to_5 = sum(1 for ep in successful_epochs if 2 <= ep <= 5)
group_more_than_5 = sum(1 for ep in successful_epochs if ep > 5)
not_converged = 100 - len(successful_epochs)

print(f"Converged in < 2 epochs: {group_less_than_2}")
print(f"Converged in 2~5 epochs: {group_2_to_5}")
print(f"Converged in > 5 epochs: {group_more_than_5}")
print(f"Failed to converge within {N_EPOCHS} epochs: {not_converged}")

# 簡單列出幾個失敗 seed 的初始權重，方便你檢查
if not_converged > 0:
    print("\n=== Sample of Failed Seeds' Initial Weights ===")
    shown = 0
    for rec in init_records:
        if rec["converged"] == 0:
            print(f"\n--- FAILED SEED: {rec['seed']} ---")
            print(f"in0->hid: {rec['w_in0_hid']:.4f}, in1->hid: {rec['w_in1_hid']:.4f}")
            print(f"in0->Sum: {rec['w_in0_sum']:.4f}, in1->Sum: {rec['w_in1_sum']:.4f}")
            print(
                f"in0->Carry: {rec['w_in0_carry']:.4f}, in1->Carry: {rec['w_in1_carry']:.4f}"
            )
            print(
                f"hid->Sum: {rec['w_hid_sum']:.4f}, hid->Carry: {rec['w_hid_carry']:.4f}"
            )
            print(f"Epoch to converge (N+1=fail): {rec['epoch']}")
            shown += 1
            if shown >= 5:
                break
