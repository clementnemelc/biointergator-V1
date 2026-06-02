import json
import os
import sys
import time
import random
import numpy as np

# Enforce UTF-8 output on Windows terminal
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import functions from SNN engine
from functions import *

# ============================================================
# 1. 超參數與實驗配置
# ============================================================
THRESHOLD   = 0.5
TICKS       = 15
LTP         = 0.04
LTD         = 0.25

TOTAL_EPOCHS = 300       # 運行 300 個 epoch 以觀察完整收斂趨勢
INIT_W_RANGE = (0.40, 0.70)

N_IN  = 9
N_OUT = 4

# ============================================================
# 2. 載入 9-to-X 的過濾真值表 (動態適應過濾後的單一或多個輸出神經元)
# ============================================================
def load_spikes_truth_table():
    json_path = "layer6_layer8_spikes_truth_table.json"
    if not os.path.exists(json_path):
        print(f"[Error] Spikes truth table file not found: {json_path}")
        sys.exit(1)
        
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
        
    layer6_labels = ["g6", "p6", "g5", "p5", "g4", "p4", "g3", "p3", "g2"]
    
    # 動態取得 layer8_spikes 中實際存在的所有標籤
    if not records:
        print("[Error] JSON records are empty.")
        sys.exit(1)
        
    layer8_labels = sorted(list(records[0]["layer8_spikes"].keys()))
    print(f"💡 偵測到 Layer 8 實際輸出標籤: {layer8_labels}")
    
    table = []
    for rec in records:
        inp = [rec["layer6_spikes"][lbl] for lbl in layer6_labels]
        out = [rec["layer8_spikes"][lbl] for lbl in layer8_labels]
        table.append((inp, out))
        
    return table, len(layer8_labels)

TRUTH_TABLE, N_OUT = load_spikes_truth_table()

# ============================================================
# 3. 網絡建立 / 狀態重置 / 單次試驗
# ============================================================
def fully_connect_with_delay(from_layer, to_layer, weight_range, delay=1):
    for src in from_layer:
        for tgt in to_layer:
            connect_modified([src], [tgt], delay=delay, weight_range=weight_range)


def build_initial_network():
    clear_synapses()
    T = THRESHOLD
    inp  = create_layer(N_IN,  name="Input",  threshold_range=(T, T), excitatory=True)
    outp = create_layer(N_OUT, name="Output", threshold_range=(T, T), excitatory=True)
    fully_connect_with_delay(inp, outp, weight_range=INIT_W_RANGE, delay=1)
    return [inp, outp]


def reset_state(net):
    for layer in net:
        for n in layer:
            n["state"].update({"v": 0.0, "spike": 0.0})
            n["input_buffer"] = [0.0] * len(n["input_buffer"])
            n["refrac_abs"] = 0
            n["refrac_rel"] = 0


def run_trial(net, inp_vec):
    reset_state(net)
    inject_input(net[0], inp_vec)
    fired_set = set()
    output_fired = [False] * N_OUT
    for _ in range(TICKS):
        step(net)
        for layer in net:
            for n in layer:
                if n["state"]["spike"] > 0:
                     fired_set.add(id(n))
        for i, n in enumerate(net[-1]):
            if n["state"]["spike"] > 0:
                output_fired[i] = True
    return [int(f) for f in output_fired], fired_set


# ============================================================
# 4. 靜默工具
# ============================================================
class SilenceStdout:
    def __enter__(self):
        self._orig = sys.stdout
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    def __exit__(self, *_):
        sys.stdout.close()
        sys.stdout = self._orig


# ============================================================
# 5. 單次訓練跑線 (無結構演化，純 STDP)
# ============================================================
def run_stdp_test(seed, swap_ltp_ltd=False):
    random.seed(seed)
    
    network = build_initial_network()
    stdp = STDPRule(ltp=LTP, ltd=LTD, w_min=0.0, w_max=1.0)
    
    locked_outputs = set()
    
    for epoch in range(TOTAL_EPOCHS):
        epoch_fired_set = set()
        
        # ── A. 訓練階段 ──
        with SilenceStdout():
            for inp_vec, exp_out in TRUTH_TABLE:
                spikes, fired_set = run_trial(network, inp_vec)
                epoch_fired_set.update(fired_set)
                
                # 計算該位元是否正確
                correct_bits = [int(spikes[i]) == exp_out[i] for i in range(N_OUT)]
                
                # 更新輸出層
                for i, out_n in enumerate(network[-1]):
                    if i in locked_outputs:
                        continue
                    
                    # 決定是否交換 LTP / LTD
                    # 正常情況下：correct -> LTP, incorrect -> LTD
                    # 交換情況下：correct -> LTD, incorrect -> LTP
                    feedback = correct_bits[i]
                    if swap_ltp_ltd:
                        feedback = not feedback
                        
                    out_n["state"]["spike"] = float(spikes[i])
                    stdp(out_n, feedback, fired_set=fired_set)
                    
        # ── Homeostatic Scaling ──
        stdp.homeostatic_scale(network[-1], epoch_fired_set, n_silent=5, scale=1.2)
        
        # ── B. 靜態驗證 (每 10 epoch 或最後一輪) ──
        should_verify = ((epoch + 1) % 10 == 0) or (epoch == TOTAL_EPOCHS - 1)
        if should_verify:
            bit_correct = [0] * N_OUT
            for inp_vec, exp_out in TRUTH_TABLE:
                spikes, _ = run_trial(network, inp_vec)
                for i in range(N_OUT):
                    if int(spikes[i]) == exp_out[i]:
                        bit_correct[i] += 1
            bit_acc = [c / len(TRUTH_TABLE) for c in bit_correct]
            
            # 鎖定機制
            for i in range(N_OUT):
                if bit_acc[i] == 1.0 and i not in locked_outputs:
                    locked_outputs.add(i)
                    
            if len(locked_outputs) == N_OUT:
                break
                
    # 最終評估
    total_correct = 0
    bit_correct = [0] * N_OUT
    for inp_vec, exp_out in TRUTH_TABLE:
        spikes, _ = run_trial(network, inp_vec)
        all_bits_ok = True
        for i in range(N_OUT):
            if int(spikes[i]) == exp_out[i]:
                bit_correct[i] += 1
            else:
                all_bits_ok = False
        if all_bits_ok:
            total_correct += 1
            
    final_bit_accs = [c / len(TRUTH_TABLE) for c in bit_correct]
    joint_acc = total_correct / len(TRUTH_TABLE)
    
    return final_bit_accs, joint_acc, epoch + 1

# ============================================================
# 6. 主程式
# ============================================================
def main():
    print("==============================================================")
    print("🔬 SNN STDP Mechanism Test: Normal vs Swapped (LTP/LTD)")
    print("   [Disabled Dynamic Neuron Mechanism, Pure STDP Only]")
    print("==============================================================")
    
    # 5 個測試隨機種子
    seeds = [42, 100, 2026, 7, 88]
    
    results = {}
    
    print(f"{'Seed':^6} | {'Mode':^8} | {'p0':^7} | {'p1':^7} | {'p2':^7} | {'p3':^7} | {'Joint Acc':^9} | {'Epochs':^6}")
    print("-" * 75)
    
    for seed in seeds:
        # 1. 正常 STDP 測試
        bit_accs_norm, joint_acc_norm, epochs_norm = run_stdp_test(seed, swap_ltp_ltd=False)
        norm_str = " | ".join([f"{a*100:4.1f}%" for a in bit_accs_norm])
        print(f"{seed:^6d} | Normal   | {norm_str} | {joint_acc_norm*100:7.1f}% | {epochs_norm:^6d}")
        
        # 2. LTP / LTD 交換測試
        bit_accs_swap, joint_acc_swap, epochs_swap = run_stdp_test(seed, swap_ltp_ltd=True)
        swap_str = " | ".join([f"{a*100:4.1f}%" for a in bit_accs_swap])
        print(f"{seed:^6d} | Swapped  | {swap_str} | {joint_acc_swap*100:7.1f}% | {epochs_swap:^6d}")
        print("-" * 75)
        
        results[str(seed)] = {
            "normal": {
                "bit_accs": bit_accs_norm,
                "joint_acc": joint_acc_norm,
                "epochs": epochs_norm
            },
            "swapped": {
                "bit_accs": bit_accs_swap,
                "joint_acc": joint_acc_swap,
                "epochs": epochs_swap
            }
        }
        
    # 儲存測試結果
    with open("stdp_swap_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("\n✨ 測試完成！詳細數據已儲存至 stdp_swap_test_results.json")

if __name__ == "__main__":
    main()
