"""
train_stress.py
壓力測試：使用比 train.py 更差的初始權重，
觀察 Soft Bound STDP + Metaplasticity + Homeostatic Scaling 能否讓網路自動收斂。

修正 XOR 競賽問題：
  Excitatory (Inp->Sum) delay = 2
  Inhibitory (Inp->Hid->Sum) delay = 1+1 = 2
  同步抵達，保證 (1,1) 被抵銷。
"""

from functions import *

# ============================================================
# 1. 超參數
# ============================================================
THRESHOLD = 0.5
TICKS = 8
N_EPOCHS = 300
LTP = 0.04
LTD = 0.15  # Balanced for faster convergence
N_STABLE = 10  # Ensure solid stability before locking
LR_DECAY = 0.5
LR_MIN = 1e-4
N_SILENT = 3
HS_SCALE = 1.5

# --- 刻意設壞的初始權重 ---
W_SUM_INIT = 0.30  # < 0.5，Sum 初始沉默
W_CARRY_INIT = 0.60  # > 0.5，Carry 初始誤射
W_HID_INIT = 0.35  # 0.35 * 2 = 0.7 > 0.5 (AND gate)
W_INH_INIT = 2.00

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

# in_put -> hid (延遲 delay=1)
fully_connect(in_put, hid, weight_range=(W_HID_INIT, W_HID_INIT))

# in_put -> Sum (out_put_0): 增加延遲到 delay=2，等待抑制訊號同步抵達
connect_modified(
    in_put,
    out_put,
    from_idx=0,
    to_idx=0,
    weight_range=(W_SUM_INIT, W_SUM_INIT),
    delay=2,
)
connect_modified(
    in_put,
    out_put,
    from_idx=1,
    to_idx=0,
    weight_range=(W_SUM_INIT, W_SUM_INIT),
    delay=2,
)

# in_put -> Carry (out_put_1): 延遲維持 1
connect_modified(
    in_put,
    out_put,
    from_idx=0,
    to_idx=1,
    weight_range=(W_CARRY_INIT, W_CARRY_INIT),
    delay=1,
)
connect_modified(
    in_put,
    out_put,
    from_idx=1,
    to_idx=1,
    weight_range=(W_CARRY_INIT, W_CARRY_INIT),
    delay=1,
)

# hid -> out_put (延遲 1)
fully_connect(hid, out_put, weight_range=(W_INH_INIT, W_INH_INIT))

print(f"\n=== Stress Test: Sum_init={W_SUM_INIT} Carry_init={W_CARRY_INIT} ===\n")

# ============================================================
# 3. STDP + Metaplasticity
# ============================================================
stdp = STDPRule(ltp=LTP, ltd=LTD, w_min=0.0, w_max=1.0)

acc_hist = []
w0_hist = []  # Sum Weight history
w1_hist = []  # Carry Weight history


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
# 4. 訓練循環
# ============================================================
streak = 0

for epoch in range(N_EPOCHS):
    n_correct = 0
    bit_errors = 0
    epoch_fired = set()

    for (A, B), (exp_s, exp_c) in zip(TRUTH_TABLE, EXPECTED):
        sf, cf, fired_set = run_trial(network, A, B)
        epoch_fired |= fired_set

        sum_ok = int(sf) == exp_s
        carry_ok = int(cf) == exp_c
        if sum_ok and carry_ok:
            n_correct += 1
        bit_errors += (0 if sum_ok else 1) + (0 if carry_ok else 1)

        out_put[0]["state"]["spike"] = float(sf)
        out_put[1]["state"]["spike"] = float(cf)
        stdp.apply_all(out_put, [sum_ok, carry_ok], fired_set=fired_set)

    acc = n_correct / len(TRUTH_TABLE)
    acc_hist.append(acc)

    # 記錄權重以便診斷
    w_s = out_put[0]["in_conns"][0]["w"]
    w_c = out_put[1]["in_conns"][0]["w"]
    w0_hist.append(w_s)
    w1_hist.append(w_c)

    # Homeostatic Scaling
    woken = stdp.homeostatic_scale(
        out_put, epoch_fired, n_silent=N_SILENT, scale=HS_SCALE
    )
    hs_msg = f"  [HS] woke: {woken}" if woken else ""

    # Metaplasticity
    if acc == 1.0:
        streak += 1
        if streak >= N_STABLE:
            stdp.ltp = max(LR_MIN, stdp.ltp * LR_DECAY)
            stdp.ltd = max(LR_MIN, stdp.ltd * LR_DECAY)
            streak = 0
            meta_msg = f"  [meta] ltp={stdp.ltp:.5f} ltd={stdp.ltd:.5f}"
        else:
            meta_msg = ""
    else:
        streak = 0
        meta_msg = ""

    if epoch % 10 == 0 or acc < 1.0 or meta_msg or hs_msg:
        tag = " OK" if acc == 1.0 else " NG"
        print(
            f"[Epoch {epoch:>3}] acc={acc:.2f} Err={bit_errors} SumW={w_s:.3f} CaryW={w_c:.3f}{tag}{hs_msg}{meta_msg}"
        )

# ============================================================
# 5. 最終驗證
# ============================================================
print("\n=== Final Truth Table ===")
for (A, B), (exp_s, exp_c) in zip(TRUTH_TABLE, EXPECTED):
    sf, cf, _ = run_trial(network, A, B)
    ok = (int(sf) == exp_s) and (int(cf) == exp_c)
    tag = "OK" if ok else "NG"
    print(
        f"  [{tag}] ({A},{B}) -> Sum={int(sf)}(exp {exp_s})  Carry={int(cf)}(exp {exp_c})"
    )

# ============================================================
# 6. 保存圖表
# ============================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
ax1.plot(acc_hist, color="#4CAF50")
ax1.set_ylabel("Accuracy")
ax1.set_title("Stress Test Progress")
ax1.set_ylim(-0.05, 1.05)

ax2.plot(w0_hist, label="Sum Synapse W", color="#2196F3")
ax2.plot(w1_hist, label="Carry Synapse W", color="#FF5722")
ax2.axhline(THRESHOLD, color="red", linestyle="--", alpha=0.5)
ax2.set_ylabel("Weight")
ax2.legend()

plt.tight_layout()
plt.savefig("stress_test_final.png")
print("\nPlot saved: stress_test_final.png")
