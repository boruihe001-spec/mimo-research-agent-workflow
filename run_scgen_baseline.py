#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

import time
import tqdm.auto  # <--- 【加上这一行，拯救世界！】
import scanpy as sc
import numpy as np
from sklearn.model_selection import train_test_split
import scgen


# 提取均值的安全辅助函数（彻底兼容稀疏矩阵和密集矩阵）
def get_safe_mean(adata_subset):
    mean_val = adata_subset.X.mean(axis=0)
    # 无论原来是 matrix 还是 ndarray，统一转为标准 1D array
    return np.asarray(mean_val).flatten()


def main():
    start_time = time.time()
    print("============================================================")
    print("🚀 Step 1: Loading data (加载并处理数据集)...")
    print("============================================================")

    # 1. 加载数据
    adata = sc.read_h5ad('./data/perturb_processed.h5ad')
    print(f"  > 原始数据维度: {adata.shape}")

    # 2. 补齐列属性
    if 'cell_type' not in adata.obs.columns:
        adata.obs['cell_type'] = 'K562'
    adata.obs['condition'] = adata.obs['condition'].astype(str)
    adata.obs['cell_type'] = adata.obs['cell_type'].astype(str)

    # 3. 提取高变基因
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    adata = adata[:, adata.var['highly_variable']].copy()
    print(f"  > 过滤高变基因后: {adata.shape}")

    # 4. 数据集划分 (严格与你的 DeepSEM 保持一致)
    ctrl_key = 'ctrl'
    conditions = adata.obs['condition'].values
    all_perts = [p for p in np.unique(conditions) if p != ctrl_key]

    single_perts = []
    for p in all_perts:
        parts = [x.strip() for x in p.split('+')]
        if len(parts) == 1 or (len(parts) == 2 and ('ctrl' in parts[0].lower() or 'ctrl' in parts[1].lower())):
            single_perts.append(p)

    train_perts, test_perts = train_test_split(single_perts, test_size=0.2, random_state=42)
    print(f"  > 训练集扰动数: {len(train_perts)} | 测试集扰动数: {len(test_perts)}")

    train_adata = adata[adata.obs['condition'].isin(list(train_perts) + [ctrl_key])].copy()
    print(f"  > 实际投入训练的细胞数: {train_adata.shape[0]}")

    print("\n============================================================")
    print("🧠 Step 2: Training scGen (开始训练深度学习基线)...")
    print("============================================================")

    # 使用正宗的 scgen API
    scgen.SCGEN.setup_anndata(train_adata, batch_key="condition", labels_key="cell_type")
    model = scgen.SCGEN(train_adata)

    # 训练模型 (对于 baseline，跑 50 轮已经足够收敛)
    print("  > 正在使用 GPU 训练模型，大概需要 3~5 分钟，请稍候...")
    model.train(max_epochs=50, batch_size=64, early_stopping=True, early_stopping_patience=5)

    print("\n============================================================")
    print("📊 Step 3: Evaluating on Test Set (在未见扰动上评估)...")
    print("============================================================")

    # 获取 Control 组的安全均值
    ctrl_adata = adata[adata.obs['condition'] == ctrl_key]
    ctrl_mean = get_safe_mean(ctrl_adata)

    n_top = 20
    scores = []

    print("  > 开始逐个预测测试集扰动...")
    for i, p in enumerate(test_perts):
        try:
            # scGen 预测 (参数名为 celltype_to_predict)
            pred_adata = model.predict(ctrl_key=ctrl_key, stim_key=p, celltype_to_predict='K562')
            pred_mean = get_safe_mean(pred_adata)
        except Exception as e:
            print(f"    [跳过] 预测 {p} 时出错: {str(e)}")
            continue

        # 真实值
        true_cells = adata[adata.obs['condition'] == p]
        if true_cells.shape[0] < 2:
            continue
        true_mean = get_safe_mean(true_cells)

        # 计算 Delta
        pred_delta = pred_mean - ctrl_mean
        true_delta = true_mean - ctrl_mean

        # 计算 Top-20 DEG 的 Pearson 相关系数
        t20 = np.argsort(np.abs(true_delta))[-n_top:]
        c = np.corrcoef(pred_delta[t20], true_delta[t20])[0, 1]

        if not np.isnan(c):
            scores.append(c)

    final_score = np.mean(scores)

    elapsed_time = time.time() - start_time
    print("\n" + "★" * 50)
    print(f"🏆 经典深度学习基线 scGen 最终评估完成！")
    print(f"   总耗时: {elapsed_time / 60:.1f} 分钟")
    print(f"   δ-DEG20: {final_score:.4f}")
    print("★" * 50)
    print(f"【下一步动作】：")
    print(f"把 {final_score:.4f} 填进 PPT 第 8 页的对比表格里，作为【深度学习基线(scGen)】。")
    print(f"如果你的 DeepSEM(0.4679) 击败了它，你的论文逻辑就彻底封神了！")


if __name__ == '__main__':
    main()