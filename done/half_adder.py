import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import functions
import visualize

importlib.reload(functions)
importlib.reload(visualize)
from functions import *

# 1. 建立網路 (閾值統一設為 0.5)
T_VAL = 0.5
create_layer(2, name="in_put", threshold_range=(T_VAL, T_VAL), excitatory=True)
create_layer(1, name="hid", threshold_range=(T_VAL, T_VAL), excitatory=False)
create_layer(2, name="out_put", threshold_range=(T_VAL, T_VAL), excitatory=True)
network = [in_put, hid, out_put]

# 2. 定義半加器布線 (精確校準)

# --- Sum 邏輯 (A OR B 部分) ---
# 延遲設為 2，確保與抑制訊號對齊
connect_modified(
    in_put, out_put, from_idx=0, to_idx=0, weight_range=(1.0, 1.0), delay=2
)
connect_modified(
    in_put, out_put, from_idx=1, to_idx=0, weight_range=(1.0, 1.0), delay=2
)

# --- Carry 邏輯 (A AND B 部分) ---
# 修正重點：權重降到 0.4！
# 讓單一個 0.4 < 0.5 (不進位)，但兩顆加起來 0.8 > 0.5 (進位)
connect_modified(
    in_put, out_put, from_idx=0, to_idx=1, weight_range=(0.4, 0.4), delay=2
)
connect_modified(
    in_put, out_put, from_idx=1, to_idx=1, weight_range=(0.4, 0.4), delay=2
)

# --- XOR 抑制路徑 ---
# 訊號路徑: in_put (delay 1) -> hid --(delay 1)--> out_put (Total delay 2)
# 修正重點：權重降到 0.4，避免單一輸入就觸發抑制
connect_modified(in_put, hid, from_idx=0, to_idx=0, weight_range=(0.4, 0.4), delay=1)
connect_modified(in_put, hid, from_idx=1, to_idx=0, weight_range=(0.4, 0.4), delay=1)
# 抑制權重 2.0，確保能完全洗掉 Sum 的興奮訊號 (1.0+1.0=2.0)
connect_modified(hid, out_put, from_idx=0, to_idx=0, weight_range=(2.0, 2.0), delay=1)

# 3. 測試循環 (捕捉瞬時脈衝)
truth_table = [(0, 0), (0, 1), (1, 0), (1, 1)]
print("\n=== Half-Adder SNN Logic Test (Improved T-Logic) ===")

for A, B in truth_table:
    # 重置狀態
    for layer in network:
        for n in layer:
            n["state"].update({"v": 0.0, "spike": 0.0})
            n["input_buffer"] = [0.0] * 10
            n["refrac_abs"] = 0

    inject_input(in_put, [A, B])

    # 建立捕捉器，看這 6 個 tick 內有沒有發生過 spike
    sum_fired = False
    carry_fired = False

    for t in range(1, 7):
        step(network)
        if out_put[0]["state"]["spike"] > 0:
            sum_fired = True
        if out_put[1]["state"]["spike"] > 0:
            carry_fired = True

    print(
        f"Input: ({A}, {B}) -> Output: [Sum: {int(sum_fired)}, Carry: {int(carry_fired)}]"
    )

print("======================================================")

# --- 4. 視覺化連接結構 ---
visualize.plot_network(network, title="Half-Adder Neural Architecture")
