import json
import sys
import os

# Enforce UTF-8 output on Windows terminal
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import functions from current directory
from functions import *

def test_multiplier():
    print("=========================================")
    print("🚀 SNN 4-Bit Multiplier 100% Accuracy Verification")
    print("=========================================")
    
    # 1. 載入 JSON 模型
    json_path = "SNN_Lean_Model (30).json"
    if not os.path.exists(json_path):
        print(f"[Error] Model JSON file not found: {json_path}")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        model = json.load(f)

    nodes = model["nodes"]
    edges = model["edges"]

    # 2. 重建網路與突觸註冊
    clear_synapses()
    layers_dict = {}
    id_to_neuron = {}

    for n in nodes:
        ly = n["layer"]
        if ly not in layers_dict:
            layers_dict[ly] = []
        
        # 建立神經元 (使用較大的 buffer_size 50 以適應深層延遲)
        neuron = create_neuron(
            idx=n["index"],
            threshold=n["threshold"],
            label=n["label"],
            excitatory=n["excitatory"],
            buffer_size=50
        )
        layers_dict[ly].append(neuron)
        id_to_neuron[n["id"]] = neuron

    network = [layers_dict[ly] for ly in sorted(layers_dict.keys())]

    # 連接突觸 (精準對接 JSON 權重與延遲)
    for edge in edges:
        src_neuron = id_to_neuron[edge["source"]]
        tgt_neuron = id_to_neuron[edge["target"]]
        
        syn = {
            "pre": src_neuron,
            "post": tgt_neuron,
            "w": edge["weight"],
            "d": edge["delay"],
            "sign": edge["sign"],
        }
        SYNAPSE_REGISTRY.append(syn)
        src_neuron["out_conns"].append(syn)
        tgt_neuron["in_conns"].append(syn)

    print("SNN 網路拓撲精準重建完成！")
    print(f"  層數 (Layers): {len(network)}")
    print(f"  神經元總數 (Neurons): {sum(len(ly) for ly in network)}")
    print(f"  突觸總數 (Synapses): {len(SYNAPSE_REGISTRY)}")

    # ============================================================
    # 3. 測試動力學仿真與解碼
    # ============================================================
    def reset_state(net):
        for layer in net:
            for n in layer:
                n["state"].update({"v": 0.0, "spike": 0.0})
                n["input_buffer"] = [0.0] * len(n["input_buffer"])
                n["refrac_abs"] = 0
                n["refrac_rel"] = 0

    def run_snn(net, input_values, ticks=40):
        reset_state(net)
        inject_input(net[0], input_values)
        
        output_layer = net[-1]
        spiked_outputs = [False] * len(output_layer)
        
        for _ in range(ticks):
            step(net)
            for i, out_n in enumerate(output_layer):
                if out_n["state"]["spike"] > 0:
                    spkd = True
                    spiked_outputs[i] = True
                    
        return [int(s) for s in spiked_outputs]

    # 4. 精確的位元對接映射 (LSB -> MSB)
    # 輸入 A = d, c, b, a (d 為 LSB, a 為 MSB)
    # 輸入 B = D, C, B, A (D 為 LSB, A 為 MSB)
    # 輸出 P = OutL10_7 to OutL10_0 (OutL10_7 為 LSB, OutL10_0 為 MSB)
    a_order = ["d", "c", "b", "a"]
    b_order = ["D", "C", "B", "A"]
    p_order = [f"OutL10_{7-i}" for i in range(8)]

    input_nodes = network[0]
    output_nodes = sorted(network[-1], key=lambda x: x["label"])

    input_label_to_node = {n["label"]: n for n in input_nodes}
    output_label_to_node = {n["label"]: n for n in output_nodes}

    def get_input_vector(A_val, B_val):
        vec = [0.0] * 8
        for bit_idx, label in enumerate(a_order):
            bit = (A_val >> bit_idx) & 1
            node = input_label_to_node[label]
            node_idx = input_nodes.index(node)
            vec[node_idx] = float(bit)
        for bit_idx, label in enumerate(b_order):
            bit = (B_val >> bit_idx) & 1
            node = input_label_to_node[label]
            node_idx = input_nodes.index(node)
            vec[node_idx] = float(bit)
        return vec

    def decode_product_from_spikes(spikes):
        val = 0
        for bit_idx, label in enumerate(p_order):
            node = output_label_to_node[label]
            node_idx = output_nodes.index(node)
            bit = spikes[node_idx]
            val |= (bit << bit_idx)
        return val

    # 5. 執行 256 個真值表組合的全量測試
    print("\n🔍 正在驗證 4-Bit 乘法真值表的所有 256 組輸入 (0-15 x 0-15)...")
    
    correct = 0
    failures = []
    
    for A_val in range(16):
        for B_val in range(16):
            in_vec = get_input_vector(A_val, B_val)
            spikes = run_snn(network, in_vec, ticks=40)
            pred = decode_product_from_spikes(spikes)
            expected = A_val * B_val
            
            if pred == expected:
                correct += 1
            else:
                failures.append((A_val, B_val, expected, pred))
                
    accuracy = correct / 256
    
    print("\n=========================================")
    print("🏆 測試驗證完成報告")
    print("=========================================")
    print(f"  總測試組合 (Total Cases): 256")
    print(f"  正確組合 (Passed): {correct}")
    print(f"  錯誤組合 (Failed): {len(failures)}")
    print(f"  靜態準確率 (Final Accuracy): {accuracy*100:.2f}%")
    print("=========================================")
    
    if len(failures) > 0:
        print("\n❌ 錯誤明細:")
        for idx, (A_val, B_val, exp, pred) in enumerate(failures[:10]):
            print(f"  {idx+1}: {A_val} x {B_val} -> Expected {exp}, got {pred}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more errors.")
    else:
        print("\n🎉 恭喜！SNN 4-bit 乘法器已完美實現 100.0% 收斂！")

if __name__ == "__main__":
    test_multiplier()
