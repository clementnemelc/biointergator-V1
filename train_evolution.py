import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Enforce UTF-8 output on Windows terminal
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import functions from current directory
from functions import *

# ============================================================
# 1. 超參數與實驗配置
# ============================================================
THRESHOLD = 0.5
TICKS = 8
LTP = 0.04
LTD = 0.30

N_ROUNDS = 3
EPOCHS_PER_ROUND = 64

TRUTH_TABLE = [
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
]
EXPECTED = [
    (0, 0), (1, 0), (1, 0), (0, 1),
    (1, 0), (0, 1), (0, 1), (1, 1)
]

def fully_connect_with_delay(from_layer, to_layer, weight_range, delay=1):
    for src in from_layer:
        for tgt in to_layer:
            connect_modified([src], [tgt], delay=delay, weight_range=weight_range)

# ============================================================
# 2. 建立與重置網絡的輔助函數
# ============================================================
def build_initial_network():
    clear_synapses()
    T = THRESHOLD
    create_layer(3, name="Input", threshold_range=(T, T), excitatory=True)
    create_layer(2, name="Output", threshold_range=(T, T), excitatory=True)
    fully_connect_with_delay(Input, Output, weight_range=(0.30, 0.60), delay=1)
    return [Input, Output]

def reset_state(net):
    for layer in net:
        for n in layer:
            n["state"].update({"v": 0.0, "spike": 0.0})
            n["input_buffer"] = [0.0] * len(n["input_buffer"])
            n["refrac_abs"] = 0
            n["refrac_rel"] = 0

def run_trial(net, A, B, Cin):
    reset_state(net)
    inject_input(net[0], [A, B, Cin])
    sf = cf = False
    fired_set = set()
    for _ in range(TICKS):
        step(net)
        for layer in net:
            for n in layer:
                if n["state"]["spike"] > 0:
                    fired_set.add(id(n))
        if net[-1][0]["state"]["spike"] > 0:
            sf = True
        if net[-1][1]["state"]["spike"] > 0:
            cf = True
    return sf, cf, fired_set

# 用於在訓練中靜默無用日誌的工具
class SilenceStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# ============================================================
# 3. 完整自適應演化 Pipeline
# ============================================================
def run_evolution_pipeline():
    print("=========================================")
    print("🚀 啟動 SNN 自適應結構演化收斂 Pipeline")
    print("=========================================")
    
    network = build_initial_network()
    stdp = STDPRule(ltp=LTP, ltd=LTD, w_min=0.0, w_max=1.0)
    locked_outputs = set()
    
    global_sum_acc = []
    global_carry_acc = []
    global_epoch_counter = 0
    evolution_points = []
    
    Hidden = []
    
    for r in range(N_ROUNDS):
        print(f"\n[Round {r+1}/{N_ROUNDS}] 啟動演化訓練... (Epochs: {EPOCHS_PER_ROUND})")
        
        for epoch in range(EPOCHS_PER_ROUND):
            epoch_fired_set = set()
            # A. 訓練階段 (靜默 LTP/LTD verbose 日誌)
            with SilenceStdout():
                for (A, B, Cin), (exp_s, exp_c) in zip(TRUTH_TABLE, EXPECTED):
                    sf, cf, fired_set = run_trial(network, A, B, Cin)
                    epoch_fired_set.update(fired_set)
                    sum_ok = int(sf) == exp_s
                    carry_ok = int(cf) == exp_c
                    
                    # 1. 更新輸出層 (直連與已存在的隱藏下遊興奮連線)
                    if 0 not in locked_outputs:
                        Output[0]["state"]["spike"] = float(sf)
                        stdp(Output[0], sum_ok, fired_set=fired_set)
                    if 1 not in locked_outputs:
                        Output[1]["state"]["spike"] = float(cf)
                        stdp(Output[1], carry_ok, fired_set=fired_set)
                    
                    # 2. 更新解凍的隱藏層 (僅當靶向輸出尚未鎖定時)
                    for h in Hidden:
                        if not h["out_conns"]:
                            continue
                        target_out = h["out_conns"][0]["post"]
                        target_idx = target_out["idx"]
                        
                        if target_idx in locked_outputs:
                            continue
                        
                        # 信用分配：設置 spike 狀態
                        h["state"]["spike"] = 1.0 if id(h) in fired_set else 0.0
                        is_correct = sum_ok if target_idx == 0 else carry_ok
                        stdp(h, is_correct, fired_set=fired_set)
                        
            # C. 突觸恆定比例縮放 (Homeostatic Scaling)，防止神經元永久靜默死亡
            stdp.homeostatic_scale(Output, epoch_fired_set, n_silent=5, scale=1.2)
            if Hidden:
                stdp.homeostatic_scale(Hidden, epoch_fired_set, n_silent=5, scale=1.2)

            # B. 靜態驗證階段
            val_sum_correct = 0
            val_carry_correct = 0
            for (A, B, Cin), (exp_s, exp_c) in zip(TRUTH_TABLE, EXPECTED):
                sf, cf, _ = run_trial(network, A, B, Cin)
                sum_ok = int(sf) == exp_s
                carry_ok = int(cf) == exp_c
                if sum_ok:
                    val_sum_correct += 1
                if carry_ok:
                    val_carry_correct += 1
                    
            val_sum_acc = val_sum_correct / len(TRUTH_TABLE)
            val_carry_acc = val_carry_correct / len(TRUTH_TABLE)
            
            global_sum_acc.append(val_sum_acc)
            global_carry_acc.append(val_carry_acc)
            global_epoch_counter += 1
            
            # 鎖定機制
            if val_sum_acc == 1.0 and 0 not in locked_outputs:
                locked_outputs.add(0)
                print(f"🔒 [Lock] Sum 於第 {global_epoch_counter} Epoch 靜態驗證達 100%，已鎖定！")
            if val_carry_acc == 1.0 and 1 not in locked_outputs:
                locked_outputs.add(1)
                print(f"🔒 [Lock] Carry 於第 {global_epoch_counter} Epoch 靜態驗證達 100%，已鎖定！")

        # 檢查是否已完全收斂
        if len(locked_outputs) == 2:
            print(f"\n🎉 網絡在第 {global_epoch_counter} Epoch 已完美解鎖雙輸出 (100% 收斂)，提前終止訓練！")
            break
            
        # 如果未完全收斂，且不是最後一輪，則執行演化
        if r < N_ROUNDS - 1:
            print(f"\n--- 第 {r+1} 輪訓練結束，評估錯誤並執行結構演化 ---")
            sum_under = sum_over = False
            carry_under = carry_over = False
            
            for (A, B, Cin), (exp_s, exp_c) in zip(TRUTH_TABLE, EXPECTED):
                sf, cf, _ = run_trial(network, A, B, Cin)
                if int(sf) != exp_s:
                    if exp_s == 1 and int(sf) == 0:
                        sum_under = True
                    elif exp_s == 0 and int(sf) == 1:
                        sum_over = True
                if int(cf) != exp_c:
                    if exp_c == 1 and int(cf) == 0:
                        carry_under = True
                    elif exp_c == 0 and int(cf) == 1:
                        carry_over = True
                        
            neurons_to_add = []
            if sum_under and 0 not in locked_outputs:
                neurons_to_add.append(("Sum_Exc", True, Output[0]))
            if sum_over and 0 not in locked_outputs:
                neurons_to_add.append(("Sum_Inh", False, Output[0]))
            if carry_under and 1 not in locked_outputs:
                neurons_to_add.append(("Carry_Exc", True, Output[1]))
            if carry_over and 1 not in locked_outputs:
                neurons_to_add.append(("Carry_Inh", False, Output[1]))
                
            if neurons_to_add:
                evolution_points.append(global_epoch_counter)
                if not Hidden:
                    network.insert(1, Hidden)
                    # 🎯 直連延遲延展 d=1 -> d=2
                    for syn in SYNAPSE_REGISTRY:
                        if syn["pre"]["label"].startswith("Input") and syn["post"]["label"].startswith("Output"):
                            syn["d"] = 2
                            print(f"⏰ [Delay Shift] 直連突觸 {syn['pre']['label']} -> {syn['post']['label']} 延遲設為 2")
                
                for name_suffix, is_exc, target_output in neurons_to_add:
                    h_idx = len(Hidden)
                    node_name = f"Hid_{name_suffix}_{h_idx}"
                    # 抑制性隱藏元設為高閾值 0.8，興奮性設為 0.5
                    thresh = 0.5 if is_exc else 0.80
                    h_neuron = create_neuron(idx=h_idx, threshold=thresh, label=node_name, excitatory=is_exc, buffer_size=20)
                    Hidden.append(h_neuron)
                    print(f"➕ 建立隱藏元: {node_name} (興奮={is_exc}, 閾值={thresh}) ───> 🎯 靶向投影至 {target_output['label']}")
                    
                    # 上游 Input -> Hidden 全連接 (delay=1)
                    fully_connect_with_delay(Input, [h_neuron], weight_range=(0.30, 0.60), delay=1)
                    # 下游 Hidden -> target_output 靶向投影 (興奮為0.3~0.6, 抑制為1.5~2.0, delay=1)
                    w_range = (0.30, 0.60) if is_exc else (1.50, 2.00)
                    fully_connect_with_delay([h_neuron], [target_output], weight_range=w_range, delay=1)
                    
            print(f"✅ 第 {r+1} 輪結構演化完成！當前隱藏層規模: {len(Hidden)}。")

    # ============================================================
    # 4. 最終驗證與輸出結果
    # ============================================================
    print(f"\n=========================================")
    print(f"📊 最終演化訓練實驗分析")
    print(f"=========================================")
    print(f"累積總訓練 Epochs: {global_epoch_counter}")
    print(f"隱藏層最終規模: {len(Hidden)} 個神經元。")
    print(f"Sum 狀態: {'鎖定 100%' if 0 in locked_outputs else '未收斂'}")
    print(f"Carry 狀態: {'鎖定 100%' if 1 in locked_outputs else '未收斂'}")
    print(f"=========================================")

    print("\n=== 最終真值表靜態驗證 ===")
    for (A, B, Cin), (exp_s, exp_c) in zip(TRUTH_TABLE, EXPECTED):
        sf, cf, _ = run_trial(network, A, B, Cin)
        sum_ok = int(sf) == exp_s
        carry_ok = int(cf) == exp_c
        tag = "[OK]" if (sum_ok and carry_ok) else "[FAIL]"
        print(f"  {tag} ({A},{B},{Cin}) -> Sum={int(sf)} (exp {exp_s}) | Cout={int(cf)} (exp {exp_c})")

    # 5. 繪製並保存學習曲線
    plt.figure(figsize=(10, 5))
    plt.plot(range(global_epoch_counter), np.array(global_sum_acc) * 100, color="purple", linewidth=2.5, label="Sum Accuracy")
    plt.plot(range(global_epoch_counter), np.array(global_carry_acc) * 100, color="green", linewidth=2.0, linestyle="--", label="Carry Accuracy")
    for pt in evolution_points:
        plt.axvline(pt, color="red", linestyle=":", alpha=0.7, label="Evolution Point" if pt == evolution_points[0] else "")
    plt.axhline(100.0, color="grey", linestyle="-.", alpha=0.5)
    plt.title("SNN Evolving Network Accuracy (Hidden Layer Unlocked & Trained)", fontsize=12, pad=15)
    plt.xlabel("Global Epoch", fontsize=10)
    plt.ylabel("Validation Accuracy (%)", fontsize=10)
    plt.ylim(-5, 105)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    sum_img_path = "sum_evolution.png"
    plt.savefig(sum_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📈 學習曲線已保存至: {sum_img_path}")

    # 匯出 JSON 模型
    json_model_path = "evolution_model.json"
    export_network_to_json(network, json_model_path)
    print(f"✨ 最終最簡拓撲模型已匯出至: {json_model_path}")

if __name__ == "__main__":
    run_evolution_pipeline()
