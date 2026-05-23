#!/usr/bin/env python3
"""
专门为中期答辩 PPT 生成高颜值配图的脚本
1. 动态缩放坐标轴的因果矩阵热力图
2. 高颜值的消融实验对比柱状图
3. 因果网络拓扑图
"""

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ==========================================
# 1. 绘制高颜值消融实验柱状图
# ==========================================
def plot_ablation_bar_chart():
    # 数据来自你刚刚跑出的终端结果
    methods = ['KNN', 'Ridge', 'VAE-only\n(Add)', 'GNN-only\n(Non-linear)', 'SEM-only\n(Causal)', 'DeepSEM\n(Full)']
    scores = [0.3343, 0.4099, 0.4217, 0.4165, 0.4438, 0.4679]

    # 颜色分配：基线为灰色，消融变体为浅蓝，最终模型为鲜艳的红色
    colors = ['#bdc3c7', '#95a5a6', '#85c1e9', '#3498db', '#2980b9', '#e74c3c']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, scores, color=colors, edgecolor='black', linewidth=1.2)

    # 在柱子上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移3个点
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylim(0.3, 0.5)  # 截取Y轴，放大差异（非常关键的作图技巧！）
    ax.set_ylabel('Delta Pearson (Top-20 DEGs)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study & Baseline Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('ppt_fig1_ablation_bars.png', dpi=300)
    print("✅ 成功生成消融实验柱状图: ppt_fig1_ablation_bars.png")


# ==========================================
# 2. 绘制自适应刻度的因果矩阵热力图
# ==========================================
def plot_adjusted_causal_matrix():
    matrix_path = 'causal_matrix_B_full.npy'
    if not os.path.exists(matrix_path):
        print(f"⚠️ 找不到 {matrix_path}，请确认文件名。")
        return

    B = np.load(matrix_path)

    # 动态计算最大绝对值，用于设置色彩范围 (解决你说的颜色太浅的问题)
    max_val = np.max(np.abs(B))
    v_limit = max_val * 1.05  # 稍微留一点余量，比如0.09

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 图1：全矩阵
    ax = axes[0]
    im = ax.imshow(B, cmap='RdBu_r', vmin=-v_limit, vmax=v_limit)
    ax.set_title(f'Full Causal Matrix B\n(Sparse: {(np.abs(B) < 0.01).mean():.1%}, Max: {max_val:.3f})', fontsize=12,
                 fontweight='bold')
    ax.set_xlabel('Cause (j)')
    ax.set_ylabel('Effect (i)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 图2：局部放大(Top-left 30x30) 以看清结构
    n_show = 30
    ax = axes[1]
    im = ax.imshow(B[:n_show, :n_show], cmap='RdBu_r', vmin=-v_limit, vmax=v_limit)
    ax.set_title(f'Zoomed (Top-left {n_show}x{n_show})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cause (j)')
    ax.set_ylabel('Effect (i)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('ppt_fig2_causal_matrix.png', dpi=300)
    print("✅ 成功生成调整刻度后的热力图: ppt_fig2_causal_matrix.png")


# ==========================================
# 3. 绘制炫酷的因果网络图 (Top 50 边)
# ==========================================
def plot_causal_network():
    matrix_path = 'causal_matrix_B_full.npy'
    if not os.path.exists(matrix_path):
        return

    B = np.load(matrix_path)
    B_no_diag = B.copy()
    np.fill_diagonal(B_no_diag, 0)
    n = B.shape[0]

    n_edges = 50
    flat_abs = np.abs(B_no_diag).flatten()
    threshold = np.sort(flat_abs)[-n_edges]

    fig, ax = plt.subplots(figsize=(6, 6))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_pos, y_pos = np.cos(angles), np.sin(angles)

    in_strength = np.sum(np.abs(B_no_diag) >= threshold, axis=1)
    out_strength = np.sum(np.abs(B_no_diag) >= threshold, axis=0)
    node_importance = in_strength + out_strength
    active_nodes = np.where(node_importance > 0)[0]

    # 画节点
    for node in active_nodes:
        size = 30 + node_importance[node] * 40
        ax.scatter(x_pos[node], y_pos[node], s=size, c='#e0f7fa', edgecolors='#00838f', linewidth=1, zorder=3)
        ax.annotate(f'{node}', (x_pos[node], y_pos[node]), fontsize=7, ha='center', va='center', zorder=4)

    # 画边
    max_weight = np.max(np.abs(B_no_diag))
    for i in range(n):
        for j in range(n):
            if i != j and np.abs(B[i, j]) >= threshold:
                color = '#e74c3c' if B[i, j] > 0 else '#2980b9'
                alpha_val = min(np.abs(B[i, j]) / max_weight, 1.0)
                ax.annotate('', xy=(x_pos[i], y_pos[i]), xytext=(x_pos[j], y_pos[j]),
                            arrowprops=dict(arrowstyle='->', color=color, alpha=max(0.3, alpha_val), lw=1.5))

    ax.set_title(f'Latent Causal Network\n(Top {n_edges} strongest edges)', fontsize=14, fontweight='bold')
    ax.set_xlim(-1.3, 1.3);
    ax.set_ylim(-1.3, 1.3);
    ax.set_aspect('equal');
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('ppt_fig3_causal_network.png', dpi=300)
    print("✅ 成功生成因果网络拓扑图: ppt_fig3_causal_network.png")


if __name__ == '__main__':
    plot_ablation_bar_chart()
    plot_adjusted_causal_matrix()
    plot_causal_network()
    print("\n🎉 全部 PPT 配图已生成完毕！你可以直接将它们拖入 PPT 第8页。")