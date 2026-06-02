import random
import os
import sys

# Import functions from SNN engine
from functions import *
from test_stdp_swap import *

def inspect_run(swap_ltp_ltd):
    random.seed(42)
    network = build_initial_network()
    stdp = STDPRule(ltp=0.04, ltd=0.25, w_min=0.0, w_max=1.0)
    
    for epoch in range(300):
        epoch_fired_set = set()
        with SilenceStdout():
            for inp_vec, exp_out in TRUTH_TABLE:
                spikes, fired_set = run_trial(network, inp_vec)
                epoch_fired_set.update(fired_set)
                correct_bits = [int(spikes[i]) == exp_out[i] for i in range(N_OUT)]
                for i, out_n in enumerate(network[-1]):
                    feedback = correct_bits[i]
                    if swap_ltp_ltd:
                        feedback = not feedback
                    out_n["state"]["spike"] = float(spikes[i])
                    stdp(out_n, feedback, fired_set=fired_set)
        stdp.homeostatic_scale(network[-1], epoch_fired_set, n_silent=5, scale=1.2)
        
    print(f"\n==========================================")
    print(f"🔍 模式: {'交換組 (Swapped)' if swap_ltp_ltd else '正常組 (Normal)'}")
    print(f"==========================================")
    
    # 1. 印出突觸權重
    print("【最終突觸權重】")
    for syn in SYNAPSE_REGISTRY:
        print(f"  {syn['pre']['label']} -> {syn['post']['label']} : w={syn['w']:.2f}")
        
    # 2. 統計預測分佈
    zeros_count = 0
    ones_count = 0
    correct_count = 0
    
    for inp_vec, exp_out in TRUTH_TABLE:
        spikes, _ = run_trial(network, inp_vec)
        pred = spikes[0]
        exp = exp_out[0]
        
        if pred == 0:
            zeros_count += 1
        else:
            ones_count += 1
            
        if pred == exp:
            correct_count += 1
            
    print("\n【預測統計】")
    print(f"  預測為 0 的次數: {zeros_count} / 256 ({zeros_count/256*100:.1f}%)")
    print(f"  預測為 1 的次數: {ones_count} / 256 ({ones_count/256*100:.1f}%)")
    print(f"  實際正確的次數: {correct_count} / 256 ({correct_count/256*100:.1f}%)")

if __name__ == "__main__":
    inspect_run(swap_ltp_ltd=False)
    inspect_run(swap_ltp_ltd=True)
