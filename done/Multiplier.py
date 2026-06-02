"""
Multiplier.py
SNN 乘法器 (1-bit AND 邏輯)
2個輸入，1個輸出 (Sum)，延遲=0
"""

from functions import *

# 1. 超參數設定
THRESHOLD = 0.5
TICKS = 5

# 2. 建立網路結構
clear_synapses()
T = THRESHOLD

# 建立 2 個輸入神經元 (A, B)
create_layer(2, name="in_p", threshold_range=(T, T), excitatory=True)
# 建立 1 個輸出神經元 (Sum)
create_layer(1, name="out_p", threshold_range=(T, T), excitatory=True)

# 3. 建立連線 (延遲 0)
# Sum 位實作 AND 邏輯：權重 0.4
# 單個輸入 (0.4) < 0.5；兩個輸入 (0.8) > 0.5
connect_modified(in_p, out_p, from_idx=0, to_idx=0, weight_range=(0.4, 0.4), delay=1)
connect_modified(in_p, out_p, from_idx=1, to_idx=0, weight_range=(0.4, 0.4), delay=1)


# 4. 定義輔助函數與測試邏輯
def reset_state(net):
    for layer in net:
        for n in layer:
            n["state"].update({"v": 0.0, "spike": 0.0})
            n["input_buffer"] = [0.0] * len(n["input_buffer"])
            n["refrac_abs"] = 0
            n["refrac_rel"] = 0


def run_test(net, A, B):
    reset_state(net)
    inject_input(net[0], [A, B])

    sf = False  # Sum bit
    for t in range(TICKS):
        step(net)
        if net[1][0]["state"]["spike"] > 0:
            sf = True

    return 1 if sf else 0


# 5. 執行測試
network = [in_p, out_p]
print(f"=== SNN Multiplier (1-bit AND) Test | Threshold: {T} ===")
for A, B in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    sum_res = run_test(network, A, B)
    print(f"Input: ({A}, {B}) -> Output Sum: {sum_res} (Expected: {A & B})")
