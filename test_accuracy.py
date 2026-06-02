
import json
import os
from functions import (
    create_neuron, 
    run_simulation, 
    clear_synapses, 
    SYNAPSE_REGISTRY
)

def load_network_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clear_synapses()
    
    # Create layers
    max_layer = data['metadata']['layers_count']
    network = [[] for _ in range(max_layer)]
    
    id_to_neuron = {}
    
    # 1. Create all neurons
    for n_data in data['nodes']:
        layer_idx = n_data['layer']
        n = create_neuron(
            idx=n_data['index'],
            threshold=n_data['threshold'],
            label=n_data['label'],
            excitatory=n_data['excitatory'],
            buffer_size=20 # Increased buffer size for safety
        )
        network[layer_idx].append(n)
        id_to_neuron[n_data['id']] = n
    
    # Sort neurons in each layer by index just in case
    for layer in network:
        layer.sort(key=lambda x: x['idx'])
        
    # 2. Create all synapses
    for e_data in data['edges']:
        src = id_to_neuron[e_data['source']]
        tgt = id_to_neuron[e_data['target']]
        
        syn = {
            "pre": src,
            "post": tgt,
            "w": e_data['weight'],
            "d": e_data['delay'],
            "sign": e_data['sign'],
        }
        SYNAPSE_REGISTRY.append(syn)
        src["out_conns"].append(syn)
        tgt["in_conns"].append(syn)
        
    return network

def int_to_4bit(n):
    return [int(b) for b in format(n, '04b')]

def spikes_to_int(spikes):
    # spikes is a dict of label: spike_value
    # The output labels are "7", "6", ..., "0"
    val = 0
    for i in range(8):
        label = str(i)
        if spikes.get(label, 0) > 0:
            val += (2 ** i)
    return val

def test_calculator_accuracy(json_path):
    network = load_network_from_json(json_path)
    
    total = 16 * 16
    correct = 0
    results = []
    
    print(f"Testing {json_path} as 4-bit * 4-bit calculator...")
    
    for i in range(16):
        for j in range(16):
            input_a = int_to_4bit(i) # e.g. [a, b, c, d] where a is MSB? 
            # Looking at labels: "a" is InL0_0, "b" is InL0_1, "c" is InL0_2, "d" is InL0_3
            # If "a" is 2^3, then format(i, '04b') gives [a, b, c, d]
            input_b = int_to_4bit(j) # "A", "B", "C", "D"
            
            # Combine inputs: a, b, c, d, A, B, C, D
            full_input = input_a + input_b
            
            # Run simulation
            # The model has 8 layers, might need more ticks for signal to propagate
            sim_result = run_simulation(network, full_input, ticks=10)
            
            # Extract final output spikes
            # history[-1] is the last tick, but wait, 
            # a spike might happen at any tick. 
            # Let's collect all spikes across all ticks for the output layer.
            final_spikes = {}
            for snapshot in sim_result['history']:
                for label, spike in snapshot.items():
                    if label in [str(k) for k in range(8)]:
                        if spike > 0:
                            final_spikes[label] = 1.0
            
            predicted = spikes_to_int(final_spikes)
            expected = i * j
            
            if i == 0 and j == 0:
                print(f"Debug: 0 * 0 predicted {predicted}")

            if predicted == expected:
                correct += 1
            else:
                results.append((i, j, expected, predicted))
                
            if (i * 16 + j + 1) % 32 == 0:
                print(f"Progress: {i * 16 + j + 1}/{total}")
                
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    
    if results:
        print("\nSome errors (first 10):")
        for res in results[:10]:
            print(f"  {res[0]} * {res[1]} = {res[2]}, predicted {res[3]}")
    
    return accuracy

if __name__ == "__main__":
    model_path = "SNN_Lean_Model (4).json"
    test_calculator_accuracy(model_path)
