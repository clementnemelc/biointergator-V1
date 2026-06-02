# train.py — 全連接半加器 STDP 訓練腳本
# 架構：2 輸入 → 1 抑制隱藏神經元 → 2 輸出（全連接）
# 學習：結果導向型 STDP（發火+正確→LTP，發火+錯誤→LTD，沉默→不變）

import importlib

import matplotlib.pyplot as plt

import functions

importlib.reload(functions)
from functions import *

# ============================================================
# 1. 超參數
# ============================================================
THRESHOLD = 0.5
TICKS = 8
N_EPOCHS = 100
LTP = 0.04  # 發火+正確 → 強化（Soft Bound：靠近 w_max 時自動減小）
LTD = 0.30  # 發火+錯誤 → 削弱（加大，使閾值附近的 LTD 遠強於 LTP）
N_STABLE = 3  # 連續幾個 epoch 全對後觸發 metaplasticity
LR_DECAY = 0.5  # 每次觸發後 LTP/LTD 乘以此係數（縮小學習率）
LR_MIN = 1e-4  # LTP/LTD 最小值下限（避免完全歸零）

# 初始權重設計：
#   in→Sum  = 0.55：超過閾值，STDP 有機會介入（正確時 LTP 強化，錯誤時 LTD 削弱）
#   in→Carry= 0.30：低於閾值，單一輸入不觸發；兩個相加 0.60 > 0.5 才觸發
#   in→hid  = 0.40：單一不觸發 hid，兩個才觸發
W_INIT_HID = (0.40, 0.40)  # in_put → hid
W_INIT_INH = (2.00, 2.00)  # hid → out_put（抑制，固定）

TRUTH_TABLE = [(0, 0), (0, 1), (1, 0), (1, 1)]
EXPECTED = [(0, 0), (1, 0), (1, 0), (0, 1)]

# ============================================================
# 2. 建立網路
# ============================================================
clear_synapses()

T = THRESHOLD
create_layer(2, name="in_put", threshold_range=(T, T), excitatory=True)
create_layer(1, name="hid", threshold_range=(T, T), excitatory=False)
create_layer(2, name="out_put", threshold_range=(T, T), excitatory=True)
network = [in_put, hid, out_put]

# in_put → hid
fully_connect(in_put, hid, weight_range=W_INIT_HID)

# in_put → Sum (out_put_0)：初始 0.55，單一輸入可激發，STDP 得以修正
connect_modified(in_put, out_put, from_idx=0, to_idx=0, weight_range=(0.55, 0.55))
connect_modified(in_put, out_put, from_idx=1, to_idx=0, weight_range=(0.55, 0.55))

# in_put → Carry (out_put_1)：初始 0.30，低於閾值，保護 AND 邏輯
connect_modified(in_put, out_put, from_idx=0, to_idx=1, weight_range=(0.30, 0.30))
connect_modified(in_put, out_put, from_idx=1, to_idx=1, weight_range=(0.30, 0.30))

# hid → out_put（固定抑制）
fully_connect(hid, out_put, weight_range=W_INIT_INH)

print(f"\n=== Network built: {len(SYNAPSE_REGISTRY)} synapses ===")
connections_check(in_put)
connections_check(hid)

# ============================================================
# 3. 初始化 STDP 物件與工具函式
# ============================================================
stdp = STDPRule(ltp=LTP, ltd=LTD, w_min=0.0, w_max=1.0)
# Soft Bound + w_max=1.0:
#   - Sum (init 0.55): LTP 收斂到 1.0，遠高於閾值 0.5，可靠發火
#   - Carry (init 0.30): LTP 推向 0.5, LTD (當單一輸入誤射時) 推回，
#     自然平衡在閾值附近，形成 AND 邏輯的「軟性保護」


def reset_state(net):
    """重置所有神經元狀態"""
    for layer in net:
        for n in layer:
            n["state"].update({"v": 0.0, "spike": 0.0})
            n["input_buffer"] = [0.0] * len(n["input_buffer"])
            n["refrac_abs"] = 0
            n["refrac_rel"] = 0


def run_trial(net, A, B):
    """
    跑一次 trial。
    回傳 (sum_fired, carry_fired, fired_set)。
    fired_set: 該 trial 中發過火的神經元 id 集合，供 STDP pre-spike 閘控用。
    """
    reset_state(net)
    inject_input(net[0], [A, B])

    sf = cf = False
    fired_set: set = set()

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
# 4. 訓練循環
# ============================================================
acc_hist = []
w_hist = {syn_id: [] for syn_id, _ in enumerate(SYNAPSE_REGISTRY)}

print(f"\n=== STDP Training ({N_EPOCHS} epochs) ===\n")

streak = 0  # 連續全對的 epoch 數

for epoch in range(N_EPOCHS):
    n_correct = 0
    bit_errors = 0

    for (A, B), (exp_s, exp_c) in zip(TRUTH_TABLE, EXPECTED):
        sf, cf, fired_set = run_trial(network, A, B)

        # 判斷各輸出神經元是否「正確」
        sum_ok = int(sf) == exp_s
        carry_ok = int(cf) == exp_c

        if sum_ok and carry_ok:
            n_correct += 1

        bit_errors += (0 if sum_ok else 1) + (0 if carry_ok else 1)

        # --- STDP 套用邏輯 ---
        out_put[0]["state"]["spike"] = float(sf)
        out_put[1]["state"]["spike"] = float(cf)
        stdp.apply_all(out_put, [sum_ok, carry_ok], fired_set=fired_set)

    acc = n_correct / len(TRUTH_TABLE)
    acc_hist.append(acc)

    # 記錄突觸權重歷史
    for syn_id, syn in enumerate(SYNAPSE_REGISTRY):
        w_hist[syn_id].append(syn["w"])

    # --- Metaplasticity ---
    if acc == 1.0:
        streak += 1
        if streak >= N_STABLE:
            # 連續全對 N_STABLE 次 → 縮小學習率
            stdp.ltp = max(LR_MIN, stdp.ltp * LR_DECAY)
            stdp.ltd = max(LR_MIN, stdp.ltd * LR_DECAY)
            streak = 0  # 重置計數，下一輪重新累積
            meta_msg = f"  [meta] ltp={stdp.ltp:.5f} ltd={stdp.ltd:.5f}"
        else:
            meta_msg = ""
    else:
        streak = 0  # 準確率不滿 → 重置連勝
        meta_msg = ""

    if epoch % 5 == 0 or acc < 1.0 or meta_msg:
        tag = " ✅" if acc == 1.0 else " ❌"
        print(f"[Epoch {epoch:>3}]  acc={acc:.2f}  bit_err={bit_errors}{tag}{meta_msg}")

# ============================================================
# 5. 最終真相表驗證
# ============================================================
print("\n=== Final Truth Table ===")
for (A, B), (exp_s, exp_c) in zip(TRUTH_TABLE, EXPECTED):
    sf, cf, _ = run_trial(network, A, B)
    ok = (int(sf) == exp_s) and (int(cf) == exp_c)
    tag = "✅" if ok else "❌"
    print(
        f"  {tag} ({A},{B}) → Sum={int(sf)}(exp {exp_s})  Carry={int(cf)}(exp {exp_c})"
    )

# ============================================================
# 6. 視覺化訓練曲線 + 突觸權重變化
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 準確度曲線
ax1.plot(acc_hist, color="tab:blue", linewidth=2, label="Accuracy")
ax1.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="100%")
ax1.set_ylabel("Accuracy")
ax1.set_ylim(-0.05, 1.1)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title("STDP Training: Half-Adder")

# 突觸權重曲線
for syn_id, syn in enumerate(SYNAPSE_REGISTRY):
    label = f"{syn['pre']['label']}→{syn['post']['label']} (sign={syn['sign']:+d})"
    style = "--" if syn["sign"] < 0 else "-"
    ax2.plot(w_hist[syn_id], linestyle=style, linewidth=1.5, label=label)

ax2.set_ylabel("Synapse Weight")
ax2.set_xlabel("Epoch")
ax2.legend(fontsize=7, ncol=2)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
