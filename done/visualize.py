import matplotlib.pyplot as plt
import networkx as nx


def plot_activity(history, title="SNN Activity (Raster Plot)"):
    """
    繪製 Raster Plot
    history: 格式為 [(tick, label), ...]
    """
    if not history:
        print("No spikes detected to plot.")
        return

    ticks = [h[0] for h in history]
    labels = [h[1] for h in history]

    # 將 label 轉為 Y 軸編號
    unique_labels = sorted(list(set(labels)))
    label_to_y = {label: i for i, label in enumerate(unique_labels)}
    y_values = [label_to_y[label] for label in labels]

    plt.figure(figsize=(10, 5))
    plt.scatter(ticks, y_values, marker="|", s=1000, color="red", linewidth=3)
    plt.yticks(range(len(unique_labels)), unique_labels)
    plt.xticks(range(max(ticks) + 2))
    plt.xlabel("Ticks (Time)")
    plt.ylabel("Neuron Labels")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_network(network, title="SNN Connection Structure"):
    """
    繪製神經網路連接結構圖
    network: [layer1, layer2, ...] 其中 layer 是 neuron 字典列表
    """
    G = nx.DiGraph()
    pos = {}
    node_colors = []

    # 1. 遍歷 network 並建立節點與位置
    for i, layer in enumerate(network):
        num_neurons = len(layer)
        for j, neuron in enumerate(layer):
            label = neuron["label"]
            G.add_node(label)
            # 手動設定分層位置 (x 軸是層, y 軸是神經元順序)
            pos[label] = (i, -j + (num_neurons - 1) / 2)

            # 定義節點顏色：興奮性紅色，抑制性藍色
            node_colors.append("lightcoral" if neuron["excitatory"] else "skyblue")

    # 2. 建立連線 (Edges)
    edge_list = []
    edge_colors = []
    edge_labels = {}

    for layer in network:
        for src in layer:
            for syn in src["out_conns"]:
                src_label = src["label"]
                tgt_label = syn["post"]["label"]
                G.add_edge(src_label, tgt_label, weight=syn["w"], delay=syn["d"])
                edge_list.append((src_label, tgt_label))

                # 連線顏色：正 sign 紅色（興奮），負 sign 藍色（抑制）
                edge_colors.append("red" if syn["sign"] > 0 else "blue")
                edge_labels[(src_label, tgt_label)] = f"w:{syn['w']:.1f},d:{syn['d']}"

    # 3. 繪圖
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # 畫邊
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edge_list,
        edge_color=edge_colors,
        arrowsize=20,
        width=2,
        alpha=0.6,
        connectionstyle="arc3,rad=0.1",  # 稍微彎曲防止重疊
    )

    # 畫權重標籤
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
