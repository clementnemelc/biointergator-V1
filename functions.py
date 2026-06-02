import json
import random
import sys

from config import *

# --- 全局突觸註冊表 ---
# 每個突觸是一個字典：{ pre, post, w, d, sign }
# 所有連線的突觸均在此集中管理，方便 LTP/LTD 操作
SYNAPSE_REGISTRY: list = []


def clear_synapses():
    """清空突觸註冊表（建立新網路前呼叫）"""
    SYNAPSE_REGISTRY.clear()


def create_neuron(idx, threshold, label="None", excitatory=True, buffer_size=10):
    return {
        "idx": idx,
        "label": label,
        "threshold_base": threshold,
        "excitatory": excitatory,
        "out_conns": [],  # 存放目標突觸 dict 的引用（向外）
        "in_conns": [],  # 存放來源突觸 dict 的引用（向內，方便 LTP）
        "input_buffer": [0.0] * buffer_size,
        "refrac_abs": 0,
        "refrac_rel": 0,
        "state": {"v": 0.0, "spike": 0.0},
    }


def inject_input(layer, values):
    if len(layer) != len(values):
        print(f"⚠️ Data mismatch. Layer: {len(layer)}, Values: {len(values)}")

    for n, val in zip(layer, values):
        if val > 0:
            n["state"]["v"] = n["threshold_base"] * 2.0
            n["state"]["spike"] = 1.0
        else:
            n["state"]["v"] = 0.0
            n["state"]["spike"] = 0.0


def create_layer(
    num_neurons,
    name="L",
    threshold_range=(1.0, 1.0),
    excitatory=None,
    excitatory_prob=0.5,
    register_to_globals=True
):
    """
    建立層並視需求註冊到全域空間
    """
    L = []
    for i in range(num_neurons):
        t = random.uniform(threshold_range[0], threshold_range[1])
        is_exc = (
            excitatory
            if excitatory is not None
            else (random.random() < excitatory_prob)
        )
        n = create_neuron(idx=i, threshold=t, label=f"{name}_{i}", excitatory=is_exc)
        L.append(n)
    
    if register_to_globals:
        # 取得「呼叫者」的 globals()
        caller_globals = sys._getframe(1).f_globals
        caller_globals[name] = L

    return L


def connect_modified(
    from_layer,
    to_layer,
    connect=True,
    weight_range=(1.0, 1.0),
    from_idx=None,
    to_idx=None,
    delay=None,
):
    """
    綜合連接函數 (支援單一連線精準操作)：
    - from_idx / to_idx: 如果有數值，則只操作該對應的神經元。
    """
    # 1. 決定來源神經元列表
    sources = [from_layer[from_idx]] if from_idx is not None else from_layer

    # 2. 決定目標神經元列表
    targets = [to_layer[to_idx]] if to_idx is not None else to_layer
    # 修正：使用 id(obj) 建立集合，因為 dict 不能直接進 set
    target_ids = {id(t) for t in targets}

    if connect:
        for src in sources:
            for tgt in targets:
                # 防止重複連線（透過 post 比對）
                if any(id(syn["post"]) == id(tgt) for syn in src["out_conns"]):
                    continue

                base_w = random.uniform(weight_range[0], weight_range[1])
                actual_delay = (
                    delay if delay is not None else max(1, abs(src["idx"] - tgt["idx"]))
                )
                sign = 1 if src["excitatory"] else -1

                # 建立突觸字典，同時存入全局註冊表
                syn = {
                    "pre": src,
                    "post": tgt,
                    "w": base_w,
                    "d": actual_delay,
                    "sign": sign,
                }
                SYNAPSE_REGISTRY.append(syn)
                src["out_conns"].append(syn)  # 前突觸持有引用
                tgt["in_conns"].append(syn)  # 後突觸持有引用
    else:
        # 斷開邏輯：從 out_conns / in_conns 移除，也從 SYNAPSE_REGISTRY 移除
        for src in sources:
            to_remove = [
                syn for syn in src["out_conns"] if id(syn["post"]) in target_ids
            ]
            for syn in to_remove:
                src["out_conns"].remove(syn)
                syn["post"]["in_conns"].remove(syn)
                if syn in SYNAPSE_REGISTRY:
                    SYNAPSE_REGISTRY.remove(syn)
            print(f"Cleared {len(to_remove)} connection(s) from {src['label']}.")


def fully_connect(from_layer, to_layer, weight_range=(0.5, 1.6)):
    for src in from_layer:
        for tgt in to_layer:
            if any(id(syn["post"]) == id(tgt) for syn in src["out_conns"]):
                continue

            base_w = random.uniform(weight_range[0], weight_range[1])
            d = abs(src["idx"] - tgt["idx"])
            sign = 1 if src["excitatory"] else -1

            syn = {"pre": src, "post": tgt, "w": base_w, "d": d, "sign": sign}
            SYNAPSE_REGISTRY.append(syn)
            src["out_conns"].append(syn)
            tgt["in_conns"].append(syn)


def connections_check(layer):
    """列印一層的所有輸出突觸資訊"""
    if not layer:
        print("Empty layer.")
        return
    name_prefix = layer[0]["label"].split("_")[0]
    print(f"--- {name_prefix} Connection Check ---")
    for n in layer:
        targets = [
            f"{syn['post']['label']}(w:{syn['w']:.2f}, d:{syn['d']}, sign:{syn['sign']:+d})"
            for syn in n["out_conns"]
        ]
        print(f"{n['label']} -> " + (", ".join(targets) if targets else "None"))
    print("=" * 30)


def get_active_neurons(network):
    """
    收集這一輪「有發 spike」的 neuron。
    回傳 list: [(src_neuron), ...] 只保留物件就好。
    """
    active = []
    for layer in network:
        for n in layer:
            if n["state"]["spike"] > 0.0:
                active.append(n)
    return active


def deliver_spikes_event_driven(active_neurons):
    """
    只針對這一輪有 spike 的 neuron，把輸出寫入目標的 input_buffer。
    符號由突觸的 sign 欄位決定（+1 興奮, -1 抑制）。
    """
    for src in active_neurons:
        for syn in src["out_conns"]:
            buf = syn["post"]["input_buffer"]
            idx = max(0, min(syn["d"] - 1, len(buf) - 1))
            buf[idx] += syn["sign"] * syn["w"]


def advance_input_buffer_and_get_in(neuron):
    """
    對單一 neuron：
    1. 把 input_buffer 往前推一格
    2. 在推的過程對還在路上的訊號做 leak
    3. 回傳本 tick 抵達 soma 的總輸入 I_in
    """
    buf = neuron["input_buffer"]
    if not buf:
        return 0.0

    arriving = buf[0]  # 這一格代表「本 tick 抵達 soma 的 input」

    # 往前推：buffer[1] -> buffer[0], buffer[2] -> buffer[1], ...
    for i in range(1, len(buf)):
        # leak：在路上訊號衰減一點
        buf[i - 1] = buf[i]

    buf[-1] = 0.0  # 最後一格清空，給未來新事件用

    return arriving


def leak_voltage(neuron):
    v = neuron["state"]["v"]
    if v < 0.0:
        # 負電位往 0 回升（再極化 → 靜止）
        v = min(0.0, v + V_NEG_LEAK)
    else:
        # 正電位往 0 下降
        v *= V_POS_LEAK
    neuron["state"]["v"] = v


def update_neurons_with_refrac_and_reset(network):
    """
    對整個 network：
    1. 從 input_buffer 推進並取得本 tick 的 I_in
    2. 積分輸入到膜電位 v
    3. 考慮絕對與相對不反應期決定 eff_threshold
    4. 決定是否發火，若發火則 v 設為負值並啟動不反應期
    5. 只有未發火的神經元才進行膜電位衰減
    """
    for layer in network:
        for n in layer:
            # 1. 推進輸入緩衝區，取得本 tick input
            I_in = advance_input_buffer_and_get_in(n)
            n["state"]["v"] += I_in

            # 2. 絕對不反應期：完全不能發火
            if n["refrac_abs"] > 0:
                n["refrac_abs"] -= 1
                n["state"]["spike"] = 0.0
                continue

            # 3. 決定本 tick 使用的 threshold（相對不反應期）
            if n["refrac_rel"] > 0:
                eff_threshold = n["threshold_base"] * REL_ALPHA
                n["refrac_rel"] -= 1
            else:
                eff_threshold = n["threshold_base"]

            # 4. 判斷是否發火
            if n["state"]["v"] >= eff_threshold:
                n["state"]["spike"] = 1.0
                n["state"]["v"] = V_RESET_NEG
                n["refrac_abs"] = ABS_TICKS
                n["refrac_rel"] = REL_TICKS
            else:
                # 只有未發火才進行膜電位衰減
                n["state"]["spike"] = 0.0
                leak_voltage(n)


def step(network):
    """
    完整的一個 tick
    """
    import config

    config.GLOBAL_TICK_COUNT += 1

    # --- 1. 兩波處理：抑制先行，興奮後至 ---

    # --- 2+3. 兩波處理：抑制先行，興奮後至 ---
    # 原理：抑制性神經元的 spike 優先寫入 buffer，
    # 使同一 tick 內興奮性神經元讀取 buf[0] 時
    # 能看到 delay=0 的抑制訊號（同 tick 取消效果）。

    # Phase A：抑制性神經元更新 + 立即傳遞
    inhib_pseudo = [[n for n in layer if not n["excitatory"]] for layer in network]
    update_neurons_with_refrac_and_reset(inhib_pseudo)
    inhib_active = get_active_neurons(inhib_pseudo)
    deliver_spikes_event_driven(inhib_active)

    # Phase B：興奮性神經元更新 + 傳遞
    excit_pseudo = [[n for n in layer if n["excitatory"]] for layer in network]
    update_neurons_with_refrac_and_reset(excit_pseudo)
    excit_active = get_active_neurons(excit_pseudo)
    deliver_spikes_event_driven(excit_active)


# =============================================================
# 可塑性規則：結果導向型 STDP
# =============================================================
class STDPRule:
    """
    結果導向型 STDP 函式物件。

    規則（僅對興奮性輸入突觸生效）：
      - 神經元發火 + 結果正確 → LTP：上游突觸權重增加（路徑強化）
      - 神經元發火 + 結果錯誤 → LTD：上游突觸權重減少（路徑削弱）
      - 神經元沉默            → 不變

    用法範例：
        stdp = STDPRule(ltp=0.05, ltd=0.03)
        stdp(output_neuron, correct=True)   # 套用到單一目標神經元
        stdp.apply_all(out_layer, correct_flags)  # 套用到整個輸出層
    """

    def __init__(
        self,
        ltp: float = 0.05,
        ltd: float = 0.03,
        w_min: float = 0.0,
        w_max: float = 10.0,
    ):
        """
        Parameters
        ----------
        ltp   : float — LTP 時每個上游突觸的權重增量
        ltd   : float — LTD 時每個上游突觸的權重減量
        w_min : float — 突觸權重下限（避免翻負）
        w_max : float — 突觸權重上限（防止爆炸）
        """
        self.ltp = ltp
        self.ltd = ltd
        self.w_min = w_min
        self.w_max = w_max
        self._history: list = []  # 記錄每次更新事件
        self._silent_counts: dict = {}  # 初始化靜默計數器

    # ------------------------------------------------------------------
    def __call__(
        self,
        neuron: dict,
        correct: bool,
        fired_set: set | None = None,
        freeze_ltp: bool = False,
    ) -> None:
        """
        對單一輸出神經元套用 STDP。

        Parameters
        ----------
        neuron    : dict       — 目標輸出神經元
        correct   : bool       — 本 trial 輸出是否正確
        fired_set : set | None — trial 中發過火的神經元 id 集合
                                 （None 則回落讀取目前 spike 狀態）
        """
        if neuron["state"]["spike"] <= 0:
            return  # 沉默 → 不變

        for syn in neuron["in_conns"]:
            # 只調整興奮性輸入，解凍隱藏層神經元相關突觸的學習限制
            if syn["sign"] < 0:
                continue

            # --- 信用分配閘控 ---
            # 優先使用 fired_set（trial 全程紀錄），避免讀到過期 spike 狀態
            if fired_set is not None:
                if id(syn["pre"]) not in fired_set:
                    continue
            else:
                if syn["pre"]["state"]["spike"] <= 0:
                    continue

            # 判斷是否為隱藏層神經元，以採用溫和穩定的學習率（解決信度分配噪聲）
            is_hidden = neuron["label"].startswith("Hid")
            effective_ltp = self.ltp * 0.5 if is_hidden else self.ltp
            effective_ltd = self.ltd * 0.15 if is_hidden else self.ltd

            if correct:
                if not freeze_ltp:
                    # Soft Bound LTP: closer to w_max, smaller increment (log convergence)
                    delta = effective_ltp * (self.w_max - syn["w"])
                    syn["w"] += delta
                    print(
                        "LTP",
                        effective_ltp,
                        syn["pre"]["label"],
                        "->",
                        syn["post"]["label"],
                        syn["w"],
                    )
                event = "LTP"
            else:
                # Soft Bound LTD: closer to w_min, smaller decrement (log convergence)
                delta = effective_ltd * (syn["w"] - self.w_min)
                syn["w"] -= delta
                print(
                    "LTD",
                    effective_ltd,
                    syn["pre"]["label"],
                    "->",
                    syn["post"]["label"],
                    syn["w"],
                )
                event = "LTD"
            # float safety clamp and precision constraint
            syn["w"] = round(max(self.w_min, min(self.w_max, syn["w"])), 2)

            self._history.append(
                {
                    "event": event,
                    "pre": syn["pre"]["label"],
                    "post": syn["post"]["label"],
                    "new_w": syn["w"],
                }
            )

    # ------------------------------------------------------------------
    def apply_all(
        self,
        layer: list,
        correct_flags: list[bool],
        fired_set: set | None = None,
        freeze_ltp: bool = False,
    ) -> None:
        """
        對整個輸出層的神經元套用 STDP。

        Parameters
        ----------
        layer         : list       — 輸出層神經元列表
        correct_flags : list[bool] — 與 layer 對應，標示每個神經元本 trial 是否正確
        fired_set     : set | None — 本次參與發火的神經元集合
        freeze_ltp    : bool       — 是否凍結 LTP（當網路已經穩定時使用）
        """
        for neuron, is_corr in zip(layer, correct_flags):
            self(neuron, correct=is_corr, fired_set=fired_set, freeze_ltp=freeze_ltp)

    # ------------------------------------------------------------------
    def homeostatic_scale(
        self,
        output_layer: list,
        epoch_fired_set: set,
        n_silent: int = 5,
        scale: float = 1.2,
    ) -> list[str]:
        """
        突觸強度稱自動縮放（Homeostatic Synaptic Scaling）。

        若某輸出神經元已連續 n_silent 個 epoch 完全沉默，
        則將其所有興奮性 in_conns 乘以 scale，使它更容易發火，
        讓 STDP 有機會介入。

        Parameters
        ----------
        output_layer    : list — 需要監控的神經元列表（通常是輸出層）
        epoch_fired_set : set  — 整個 epoch 中發過火的神經元 id 集合
        n_silent        : int  — 觸發縮放所需的連續靜默 epoch 數
        scale           : float — 縮放係數（> 1 表示增強）

        Returns
        -------
        list[str] — 被縮放的神經元 label 列表（供外部記錄）
        """
        if not hasattr(self, "_silent_counts"):
            self._silent_counts: dict = {}

        triggered = []
        for neuron in output_layer:
            nid = id(neuron)
            if nid in epoch_fired_set:
                # 本 epoch 有發火 → 重置計數
                self._silent_counts[nid] = 0
            else:
                # 本 epoch 完全沉默 → 計數 +1
                self._silent_counts[nid] = self._silent_counts.get(nid, 0) + 1

            if self._silent_counts[nid] >= n_silent:
                # 觸發 Homeostatic 縮放
                for syn in neuron["in_conns"]:
                    if syn["sign"] > 0:  # 只縮放興奮性突觸
                        syn["w"] = round(min(self.w_max, syn["w"] * scale), 2)
                self._silent_counts[nid] = 0  # 重置，避免連續觸發
                triggered.append(neuron["label"])

        return triggered

    # ------------------------------------------------------------------
    def report(self, last_n: int = 10) -> None:
        """印出最近 n 筆更新紀錄"""
        print(f"=== STDP History (last {last_n}) ===")
        for ev in self._history[-last_n:]:
            print(
                f"  [{ev['event']}] {ev['pre']} -> {ev['post']}  new_w={ev['new_w']:.4f}"
            )

    # ------------------------------------------------------------------
    def clear_history(self) -> None:
        """清空更新紀錄"""
        self._history.clear()


def export_network_to_json(network, filename="model_export.json"):
    """
    將目前的網路打包成輕量化且具備「座標型 ID」的 JSON (如 L0_0, L1_5)
    """
    nodes = []
    edges = []
    
    # 建立一個物件到新 ID 的映射表，這能確保 ID 刷新時連線不會斷開
    obj_to_id = {}
    
    # 1. 第一輪：決定所有神經元的新 ID 並儲存節點資訊
    for l_idx, layer in enumerate(network):
        for n_idx, n in enumerate(layer):
            new_id = f"L{l_idx}_{n_idx}"
            obj_to_id[id(n)] = new_id # 使用 Python 物件記憶體位址作為 key
            
            nodes.append({
                "id": new_id,
                "label": n["label"], # 保留原始標籤供顯示
                "layer": l_idx,
                "index": n_idx,
                "type": "input" if l_idx == 0 else ("output" if l_idx == len(network)-1 else "hidden"),
                "excitatory": n["excitatory"],
                "threshold": round(n["threshold_base"], 2)
            })

    # 2. 第二輪：使用新 ID 建立連線資訊
    for s_idx, syn in enumerate(SYNAPSE_REGISTRY):
        src_id = obj_to_id.get(id(syn["pre"]))
        tgt_id = obj_to_id.get(id(syn["post"]))
        
        if src_id and tgt_id:
            edges.append({
                "id": f"e_{src_id}_to_{tgt_id}",
                "source": src_id,
                "target": tgt_id,
                "weight": round(syn["w"], 4),
                "delay": syn["d"],
                "sign": syn["sign"]
            })

    # 3. 封裝
    model_data = {
        "metadata": {
            "name": "SNN_Coordinate_Model",
            "layers_count": len(network),
            "nodes_count": len(nodes),
            "synapses_count": len(edges)
        },
        "nodes": nodes,
        "edges": edges
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=4)

    print(f"✨ 座標型 ID 模型已導出: {filename} (ID 已根據位置刷新)")
    return model_data


# =============================================================
# 高階功能：模擬與網路管理 (新增加)
# =============================================================

def reset_network_state(network):
    """
    將網路中所有神經元的電位、發火狀態與緩衝區歸零
    """
    for layer in network:
        for n in layer:
            n["state"].update({"v": 0.0, "spike": 0.0})
            n["input_buffer"] = [0.0] * len(n["input_buffer"])
            n["refrac_abs"] = 0
            n["refrac_rel"] = 0
    # print("🔄 Network state reset.")


def run_simulation(network, input_values, ticks=8):
    """
    測試輸入脈衝功能：
    執行 ticks 個時間步，並回傳完整的發火紀錄 (Spike History)
    
    回傳格式: 
    {
        "output_fired": bool,  # 輸出層最後是否至少發火一次
        "history": [           # 每一刻的所有神經元狀態 (供網頁動畫使用)
            { "neuron_label": 0/1, ... }, # Tick 1
            { "neuron_label": 0/1, ... }, # Tick 2
            ...
        ]
    }
    """
    reset_network_state(network)
    inject_input(network[0], input_values)
    
    history = []
    output_fired_at_least_once = False
    output_layer = network[-1]

    for t in range(ticks):
        step(network)
        
        # 紀錄當前所有節點的發火狀況
        tick_snapshot = {}
        for layer in network:
            for n in layer:
                tick_snapshot[n["label"]] = n["state"]["spike"]
        
        history.append(tick_snapshot)

        # 檢查輸出層
        for n in output_layer:
            if n["state"]["spike"] > 0:
                output_fired_at_least_once = True

    return {
        "output_fired": output_fired_at_least_once,
        "history": history
    }


def add_layer_to_network(network, num_neurons, name=None, threshold=0.5, excitatory=True):
    """
    動態增加 Layer 功能 (已優化唯一性命名):
    在現有網路列表的中間插入一層（通常是在 Output 之前）
    """
    # 如果沒指定名稱，根據目前層數自動生成，避免重複標籤 (Label/ID Collision)
    if name is None:
        name = f"HidL{len(network)}"
    
    new_layer = create_layer(
        num_neurons, 
        name=name, 
        threshold_range=(threshold, threshold), 
        excitatory=excitatory,
        register_to_globals=False 
    )
    
    # 插在 Output 之前 (最後一層通常是輸出)
    network.insert(-1, new_layer)
    print(f"➕ Layer '{name}' added at index {network.index(new_layer)}. (Total layers: {len(network)})")
    
    # 提醒：export_network_to_json 會在下一次導出時
    # 透過 enumerate(network) 自動重新編排所有節點的 'layer' 屬性。
    return new_layer
