import json
import os
import sys
import time
import matplotlib.pyplot as plt
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
TICKS       = 15         # 12->8 層信號傳播時間
LTP         = 0.04
LTD         = 0.25

N_ROUNDS          = 15   # 演化輪次上限
EPOCHS_PER_ROUND  = 160  # 每輪 Epoch 數 (160 epoch)
INIT_W_RANGE      = (0.40, 0.70) # 初始權重範圍

N_IN  = 9
N_OUT = 4

# ============================================================
# 2. 載入並重塑 Layer 6 到 Layer 8 的真值表
# ============================================================
def load_spikes_truth_table():
    json_path = "layer6_layer8_spikes_truth_table.json"
    if not os.path.exists(json_path):
        print(f"[Error] Spikes truth table file not found: {json_path}")
        print("請確認是否已先執行 monitor_layers_truth_table.py 以產生真值表。")
        sys.exit(1)
        
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
        
    layer6_labels = ["g6", "p6", "g5", "p5", "g4", "p4", "g3", "p3", "g2"]
    layer8_labels = ["7", "6", "5", "4"]
    
    table = []
    for rec in records:
        inp = [rec["layer6_spikes"][lbl] for lbl in layer6_labels]
        out = [rec["layer8_spikes"][lbl] for lbl in layer8_labels]
        table.append((inp, out))
        
    return table, len(layer6_labels), len(layer8_labels)

TRUTH_TABLE, _, _ = load_spikes_truth_table()

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
# 5. 演化 Pipeline (支援 5 分鐘超時終止)
# ============================================================
def run_evolution_pipeline():
    print("=========================================")
    print("🚀 SNN Layer 6 -> Layer 8 自適應演化 Pipeline")
    print(f"   Input neurons: {N_IN}  |  Output neurons: {N_OUT}")
    print(f"   Truth table entries: {len(TRUTH_TABLE)}")
    print("=========================================")

    network  = build_initial_network()
    stdp     = STDPRule(ltp=LTP, ltd=LTD, w_min=0.0, w_max=1.0)
    Hidden   = []

    locked_outputs  = set()      # 已收斂至 100% 的輸出 bit 索引
    global_acc      = [[] for _ in range(N_OUT)]   # 每個輸出 bit 的 acc 曲線
    global_epoch    = 0
    evolution_pts   = []

    start_time = time.time()
    timeout_seconds = 300.0  # 5 minutes
    timeout_triggered = False

    for r in range(N_ROUNDS):
        print(f"\n[Round {r+1}/{N_ROUNDS}] 演化訓練中... (Epochs/Round={EPOCHS_PER_ROUND})")

        for epoch in range(EPOCHS_PER_ROUND):
            # 檢測 5 分鐘超時
            if time.time() - start_time > timeout_seconds:
                print(f"\n⚠️ [Timeout] 訓練執行時間已達 5 分鐘 ({timeout_seconds}s) 硬上限！自動觸發終止...")
                timeout_triggered = True
                break

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

            # ── B. 靜態驗證 (改為每 10 次驗證一次) ──
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
                        elapsed = time.time() - start_time
                        print(f"🔒 [Lock] Output bit p{i} 於 Epoch {global_epoch + 1} 達 100%！時間: {elapsed:.1f}s (locked={sorted(locked_outputs)})")
            else:
                # 若本次不驗證，重用上一次的準確率（若是第一個 Epoch 則為 0）
                if global_epoch == 0:
                    bit_acc = [0.0] * N_OUT
                else:
                    bit_acc = [global_acc[i][-1] for i in range(N_OUT)]

            for i in range(N_OUT):
                global_acc[i].append(bit_acc[i])
            global_epoch += 1

            # 進度列印 (每 20 epoch 且當前發生驗證時)
            if (epoch % 20 == 19 and should_verify) or len(locked_outputs) == N_OUT:
                acc_str = " ".join([f"p{i}:{bit_acc[i]*100:.0f}%" for i in range(N_OUT)])
                elapsed = time.time() - start_time
                print(f"  [E{global_epoch:>4}] Locked={len(locked_outputs)}/8 | {acc_str} | 時間: {elapsed:.1f}s")

        if timeout_triggered:
            break

        # 全部收斂 → 提前終止
        if len(locked_outputs) == N_OUT:
            print(f"\n🎉 所有 8 個輸出 bit 均已完美收斂！總 Epoch: {global_epoch}")
            break

        # ── 結構演化：為未收斂 bit 加入隱藏元 ──
        if r < N_ROUNDS - 1:
            print(f"\n--- Round {r+1} 結束，執行結構演化 ---")

            # 檢查隱藏神經元個數是否已經達到上限 10
            if len(Hidden) == 10:
                print("\n🎯 [Limit Reach] 隱藏神經元總數已達上限 10 個，且已額外完成一輪完整訓練！程式自動停止並輸出報告...")
                break

            # 分析每個未鎖定 bit 的錯誤類型
            under = set()   # 需要更多激發（期望=1，實際=0）
            over  = set()   # 需要更多抑制（期望=0，實際=1）
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
                    if len(Hidden) >= 10:
                        print("  ⚠️ [Limit] 隱藏神經元數已達 10 個上限，停止新增 Exc 隱藏元！")
                        break
                    h_idx  = len(Hidden)
                    h_name = f"Hid_Exc_p{i}_{h_idx}"
                    h_n    = create_neuron(idx=h_idx, threshold=0.5, label=h_name, excitatory=True, buffer_size=30)
                    Hidden.append(h_n)
                    fully_connect_with_delay(network[0], [h_n], weight_range=INIT_W_RANGE, delay=1)
                    fully_connect_with_delay([h_n], [target_out], weight_range=(0.30, 0.60), delay=1)
                    print(f"  ➕ Exc hidden → p{i}: {h_name}")
                    neurons_added += 1

                if i in over:
                    if len(Hidden) >= 10:
                        print("  ⚠️ [Limit] 隱藏神經元數已達 10 個上限，停止新增 Inh 隱藏元！")
                        break
                    h_idx  = len(Hidden)
                    h_name = f"Hid_Inh_p{i}_{h_idx}"
                    h_n    = create_neuron(idx=h_idx, threshold=0.80, label=h_name, excitatory=False, buffer_size=30)
                    Hidden.append(h_n)
                    fully_connect_with_delay(network[0], [h_n], weight_range=INIT_W_RANGE, delay=1)
                    fully_connect_with_delay([h_n], [target_out], weight_range=(1.50, 2.00), delay=1)
                    print(f"  ➕ Inh hidden → p{i}: {h_name}")
                    neurons_added += 1

            if neurons_added > 0:
                evolution_pts.append(global_epoch)
                if first_hidden:
                    # 把所有直連突觸延遲從 1 改成 2
                    for syn in SYNAPSE_REGISTRY:
                        pre_lbl  = syn["pre"]["label"]
                        post_lbl = syn["post"]["label"]
                        if pre_lbl.startswith("Input") and post_lbl.startswith("Output"):
                            syn["d"] = 2
                    print(f"  ⏰ 直連延遲已調整為 d=2")
                    network.insert(1, Hidden)

            print(f"  ✅ 結構演化完成！隱藏層規模: {len(Hidden)} 個神經元")

    # ============================================================
    # 6. 最終驗證與匯報
    # ============================================================
    print("\n=========================================")
    print("📊 最終演化擬合與收斂情況匯報")
    print("=========================================")
    
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

    final_accuracy = total_correct / len(TRUTH_TABLE)
    elapsed_total = time.time() - start_time
    
    print(f"  總執行時間 (Total Time): {elapsed_total:.2f} 秒")
    print(f"  總 Epoch 數 (Total Epochs): {global_epoch}")
    print(f"  已鎖定 bits 數: {len(locked_outputs)} / {N_OUT}")
    print(f"  最終完美配對: {total_correct} / {len(TRUTH_TABLE)} 筆 (全位元完全正確)")
    print(f"  全位元準確率 (Overall Accuracy): {final_accuracy*100:.2f}%")
    print(f"  隱藏層神經元總數 (Hidden Neurons): {len(Hidden)}")
    
    print("\n  [各輸出 Bit 擬合 Acc 詳情]")
    for i in range(N_OUT):
        acc = bit_correct[i] / len(TRUTH_TABLE)
        status = "🔒 LOCKED 100%" if i in locked_outputs else f"⚠️ UNCONVERGED ({acc*100:.1f}%)"
        print(f"    - Bit p{i}: {status}")
    print("=========================================")

    # ============================================================
    # 7. 繪製學習曲線
    # ============================================================
    colors = plt.cm.tab10(np.linspace(0, 1, N_OUT))
    plt.figure(figsize=(12, 6))
    for i in range(N_OUT):
        plt.plot(global_acc[i], color=colors[i], linewidth=1.5, label=f"Bit p{i}")
    for pt in evolution_pts:
        plt.axvline(pt, color="red", linestyle=":", alpha=0.6,
                    label="Structure Evolution" if pt == evolution_pts[0] else "")
    plt.axhline(1.0, color="grey", linestyle="-.", alpha=0.4)
    plt.title("SNN Adaptive Evolution — Layer 6 to Layer 8 Spike Mapping", fontsize=13)
    plt.xlabel("Global Epoch")
    plt.ylabel("Validation Accuracy")
    plt.ylim(-0.05, 1.08)
    plt.legend(loc="lower right", ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plot_path = "layer6_to_layer8_evolution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📈 學習曲線已儲存: {plot_path}")

    # ============================================================
    # 8. 匯出新模型
    # ============================================================
    model_path = "layer6_to_layer8_evolved_model.json"
    export_network_to_json(network, model_path)
    print(f"✨ 最終演化模型已匯出: {model_path}")
    print("=========================================")

    return final_accuracy

if __name__ == "__main__":
    run_evolution_pipeline()
