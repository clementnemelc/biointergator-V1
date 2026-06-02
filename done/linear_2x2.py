"""
linear_2x2.py
實作 2x2 線性層 SNN: y = W*x
W = [[1, 2],
     [3, 1]]
使用時間延遲編碼（Temporal Delay Coding）來表示大於 1 的整數權重，
以適應神經元的絕對不反應期（ABS_TICKS=2），避免同一個 tick 接收太多峰值被重置截斷。
"""

import config
from functions import *

# 設定
THRESHOLD = 1.0
TICKS = 15  # 模擬時間足夠讓延遲的 spike 傳遞完畢


def build_linear_layer():
    clear_synapses()
    # 建立輸入和輸出層
    in_l = create_layer(
        2, name="lin", threshold_range=(THRESHOLD, THRESHOLD), excitatory=True
    )
    out_l = create_layer(
        2, name="lout", threshold_range=(THRESHOLD, THRESHOLD), excitatory=True
    )

    # 由於 ABS_TICKS = 2，神經元發火後有 2 個 tick 不能發火。
    # 所以到達 output 的 spike 必須間隔至少 3 個 delay，例如 delay = 1, 4, 7, 10

    # 權重矩陣 W = [[1, 2], [3, 1]]
    # y0 = 1*x0 + 2*x1
    # x0 -> y0 : 1.0 (delay 1)
    connect_modified(
        in_l, out_l, from_idx=0, to_idx=0, weight_range=(3.0, 3.0), delay=1
    )
    # x1 -> y0 : 2.0 (delay 2, 5)
    connect_modified(
        in_l, out_l, from_idx=1, to_idx=0, weight_range=(3.0, 3.0), delay=2
    )
    connect_modified(
        in_l, out_l, from_idx=1, to_idx=0, weight_range=(3.0, 3.0), delay=5
    )

    # y1 = 3*x0 + 1*x1
    # x0 -> y1 : 3.0 (delay 1, 4, 7)
    connect_modified(
        in_l, out_l, from_idx=0, to_idx=1, weight_range=(3.0, 3.0), delay=1
    )
    connect_modified(
        in_l, out_l, from_idx=0, to_idx=1, weight_range=(3.0, 3.0), delay=4
    )
    connect_modified(
        in_l, out_l, from_idx=0, to_idx=1, weight_range=(3.0, 3.0), delay=7
    )
    # x1 -> y1 : 1.0 (delay 2) - interleaved
    connect_modified(
        in_l, out_l, from_idx=1, to_idx=1, weight_range=(3.0, 3.0), delay=2
    )

    return [in_l, out_l]


def reset_state(net):
    config.GLOBAL_TICK_COUNT = 0
    for layer in net:
        for n in layer:
            n["state"].update({"v": 0.0, "spike": 0.0})
            n["input_buffer"] = [0.0] * len(n["input_buffer"])
            n["refrac_abs"] = 0
            n["refrac_rel"] = 0


def run_linear_trial(net, inputs):
    reset_state(net)
    inject_input(net[0], inputs)

    # 統計 output spike 數量
    spike_counts = [0, 0]

    for _ in range(TICKS):
        step(net)
        # 統計 y0, y1 發火
        if net[-1][0]["state"]["spike"] > 0:
            spike_counts[0] += 1
        if net[-1][1]["state"]["spike"] > 0:
            spike_counts[1] += 1

    return spike_counts


if __name__ == "__main__":
    net = build_linear_layer()

    truth_table = [
        ((0, 0), [0, 0]),
        ((0, 1), [2, 1]),
        ((1, 0), [1, 3]),
        ((1, 1), [3, 4]),
    ]

    print("=== 2x2 Linear Layer Truth Table Verification ===")
    print(
        f"{'Input (x0,x1)':^15} | {'Outputs (y0, y1)':^20} | {'Expected':^15} | {'Match':^5}"
    )
    print("-" * 65)

    all_pass = True
    for (x0, x1), expected in truth_table:
        counts = run_linear_trial(net, (x0, x1))
        match = counts == expected
        if not match:
            all_pass = False
        print(
            f"({x0}, {x1})".center(15)
            + " | "
            + f"[{counts[0]}, {counts[1]}]".center(20)
            + " | "
            + f"[{expected[0]}, {expected[1]}]".center(15)
            + " | "
            + str(match).center(5)
        )

    print("-" * 65)
    print(f"Final Status: {'Passed' if all_pass else 'Failed'}")
