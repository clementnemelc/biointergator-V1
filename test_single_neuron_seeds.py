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

N_ROUNDS          = 8
EPOCHS_PER_ROUND  = 40
INIT_W_RANGE      = (0.40, 0.70)

N_IN  = 9
N_OUT = 1

# ============================================================
# 2. 載入 9-to-1 的聚焦真值表
# ============================================================
def load_spikes_truth_table():
    json_path = "layer6_layer8_spikes_truth_table.json"
    if not os.path.exists(json_path):
        print(f"[Error] Spikes truth table file not found: {json_path}")
        sys.exit(1)
        
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
        
    layer6_labels = ["g6", "p6", "g5", "p5", "g4", "p4", "g3", "p3", "g2"]
    layer8_labels = ["7"]
    
    table = []
    for rec in records:
        inp = [rec["layer6_spikes"][lbl] for lbl in layer6_labels]
        out = [rec["layer8_spikes"][lbl] for lbl in layer8_labels]
        table.append((inp, out))
        
    return table

TRUTH_TABLE = load_spikes_truth_table()

# ============================================================
# 3. 網路建立 / 狀態重置 / 單次試驗
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
# 4. 靜默工具 (用以遮蔽大量訓練日誌，保持終端機乾淨)
# ============================================================
class SilenceStdout:
    def __enter__(self):
        self._orig = sys.stdout
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    def __exit__(self, *_):
        sys.stdout.close()
        sys.stdout = self._orig


# ============================================================
# 5. 單隨機種子的演化訓練管線
# ============================================================
def run_evolution_pipeline(seed):
    # 設定隨機種子以保證可重現性
    random.seed(seed)
    
    network  = build_initial_network()
    stdp     = STDPRule(ltp=LTP, ltd=LTD, w_min=0.0, w_max=1.0)
    Hidden   = []

    locked_outputs  = set()
    global_acc      = [[] for _ in range(N_OUT)]
    global_epoch    = 0

    for r in range(N_ROUNDS):
        for epoch in range(EPOCHS_PER_ROUND):
            epoch_fired_set = set()

            # ── A. 訓練階段 ──
            with SilenceStdout():
                for inp_vec, exp_out in TRUTH_TABLE:
                    spikes, fired_set = run_trial(network, inp_vec)
                    epoch_fired_set.update(fired_set)
                    correct_bits = [int(spikes[i]) == exp_out[i] for i in range(N_OUT)]

                    # 更新輸出層
                    for i, out_n in enumerate(network[-1]):
                        if i in locked_outputs:
                            continue
                        out_n["state"]["spike"] = float(spikes[i])
                        stdp(out_n, correct_bits[i], fired_set=fired_set)

                    # 更新隱藏層
                    for h in Hidden:
                        if not h["out_conns"]:
                            continue
                        target_out = h["out_conns"][0]["post"]
                        target_idx = target_out["idx"]
                        if target_idx in locked_outputs:
                            continue
                        h["state"]["spike"] = 1.0 if id(h) in fired_set else 0.0
                        is_correct = correct_bits[target_idx]
                        stdp(h, is_correct, fired_set=fired_set)

            # ── Homeostatic Scaling ──
            stdp.homeostatic_scale(network[-1], epoch_fired_set, n_silent=5, scale=1.2)
            if Hidden:
                stdp.homeostatic_scale(Hidden, epoch_fired_set, n_silent=5, scale=1.2)

            # ── B. 靜態驗證 (每 10 次驗證一次) ──
            should_verify = ((epoch + 1) % 10 == 0) or (epoch == EPOCHS_PER_ROUND - 1) or (len(locked_outputs) == N_OUT)
            
            if should_verify:
                bit_correct = [0] * N_OUT
                for inp_vec, exp_out in TRUTH_TABLE:
                    spikes, _ = run_trial(network, inp_vec)
                    for i in range(N_OUT):
                        if int(spikes[i]) == exp_out[i]:
                            bit_correct[i] += 1
                bit_acc = [c / len(TRUTH_TABLE) for c in bit_correct]
                
                # ── 鎖定已收斂的 bits ──
                for i in range(N_OUT):
                    if bit_acc[i] == 1.0 and i not in locked_outputs:
                        locked_outputs.add(i)
            else:
                if global_epoch == 0:
                    bit_acc = [0.0] * N_OUT
                else:
                    bit_acc = [global_acc[i][-1] for i in range(N_OUT)]

            for i in range(N_OUT):
                global_acc[i].append(bit_acc[i])
            global_epoch += 1

        # 全部收斂 → 提前終止
        if len(locked_outputs) == N_OUT:
            break

        # ── 結構演化：為未收斂 bit 加入隱藏元 ──
        if r < N_ROUNDS - 1:
            # 檢查隱藏神經元個數是否已經達到上限 8
            if len(Hidden) == 8:
                break

            # 分析每個未鎖定 bit 的錯誤類型
            under = set()
            over  = set()
            for inp_vec, exp_out in TRUTH_TABLE:
                spikes, _ = run_trial(network, inp_vec)
                for i in range(N_OUT):
                    if i in locked_outputs:
                        continue
                    if exp_out[i] == 1 and spikes[i] == 0:
                        under.add(i)
                    elif exp_out[i] == 0 and spikes[i] == 1:
                        over.add(i)

            neurons_added = 0
            first_hidden  = len(Hidden) == 0

            for i in range(N_OUT):
                if i in locked_outputs:
                    continue
                target_out = network[-1][i]

                if i in under:
                    if len(Hidden) >= 8:
                        break
                    h_idx  = len(Hidden)
                    h_name = f"Hid_Exc_p{i}_{h_idx}"
                    h_n    = create_neuron(idx=h_idx, threshold=0.5, label=h_name, excitatory=True, buffer_size=30)
                    Hidden.append(h_n)
                    fully_connect_with_delay(network[0], [h_n], weight_range=INIT_W_RANGE, delay=1)
                    fully_connect_with_delay([h_n], [target_out], weight_range=(0.30, 0.60), delay=1)
                    neurons_added += 1

                if i in over:
                    if len(Hidden) >= 8:
                        break
                    h_idx  = len(Hidden)
                    h_name = f"Hid_Inh_p{i}_{h_idx}"
                    h_n    = create_neuron(idx=h_idx, threshold=0.80, label=h_name, excitatory=False, buffer_size=30)
                    Hidden.append(h_n)
                    fully_connect_with_delay(network[0], [h_n], weight_range=INIT_W_RANGE, delay=1)
                    fully_connect_with_delay([h_n], [target_out], weight_range=(1.50, 2.00), delay=1)
                    neurons_added += 1

            if neurons_added > 0:
                if first_hidden:
                    for syn in SYNAPSE_REGISTRY:
                        pre_lbl  = syn["pre"]["label"]
                        post_lbl = syn["post"]["label"]
                        if pre_lbl.startswith("Input") and post_lbl.startswith("Output"):
                            syn["d"] = 2
                    network.insert(1, Hidden)

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
    
    return final_bit_accs, joint_acc, len(Hidden)

# ============================================================
# 6. 主控程式 (遍歷 10 個隨機種子)
# ============================================================
def main():
    print("======================================================")
    print("🚀 SNN Single Neuron Study: Layer 8 '7' (10 Seeds, Max 8 Nodes)")
    print("======================================================")
    
    # 10 組獨立的隨機種子
    seeds = [42, 100, 2026, 7, 88, 999, 12345, 888, 777, 123]
    
    results = {}
    
    print(f"{'Seed':^8} | {'Output label=7 Acc':^22} | {'Hidden Nodes':^14} | {'Time(s)':^8}")
    print("-" * 65)
    
    t_start = time.time()
    
    for seed in seeds:
        t0 = time.time()
        bit_accs, joint_acc, hidden_count = run_evolution_pipeline(seed)
        t_elapsed = time.time() - t0
        
        results[str(seed)] = {
            "bit_accs": bit_accs,
            "joint_acc": joint_acc,
            "hidden_count": hidden_count,
            "time_seconds": round(t_elapsed, 2)
        }
        
        # 輸出單一種子結果
        print(f"{seed:^8d} | {bit_accs[0]*100:19.2f}% | {hidden_count:^14d} | {t_elapsed:6.1f}s")
        
    t_total = time.time() - t_start
    print("=" * 65)
    print(f"Total verification completed in {t_total:.2f} seconds.")
    
    # 儲存結果 JSON
    output_path = "single_neuron_seed_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Successfully exported single-neuron study results to: {output_path}")

if __name__ == "__main__":
    main()
