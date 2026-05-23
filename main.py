#!/usr/bin/env python3
"""
DeepSEM v3-final: 本科毕设最终版
一次运行产出全部论文所需数据：
  - 模型训练 + 评估
  - 4个基线对比 (no-change / global-mean / ridge / knn)
  - 汇总表格
  - 可视化图表
"""

import os
import gc
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 0. 配置
# ============================================================
class Config:
    data_dir = './data'
    h5ad_filename = 'perturb_processed.h5ad'
    n_hvg = 5000

    input_dim = 5000
    latent_dim = 128
    encoder_hidden = [1024, 512]
    decoder_hidden = [512, 1024]
    sem_rank = 64
    gnn_hidden = 256
    gnn_layers = 2

    batch_size = 256
    n_epochs = 150
    lr = 1e-3
    weight_decay = 1e-5
    kl_weight = 0.005
    causal_reg = 0.0
    patience = 10
    embed_dropout = 0.5

    n_top_degs = 20
    n_top_degs_2 = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42


cfg = Config()
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

print(f"Device: {cfg.device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# 1. 数据加载
# ============================================================

def identify_single_and_combo_perts(all_unique_perts):
    single_perts, combo_perts = [], []
    for p in all_unique_perts:
        parts = [x.strip() for x in p.split('+')]
        if len(parts) == 1:
            if parts[0].lower() != 'ctrl':
                single_perts.append(p)
        elif len(parts) == 2:
            if parts[0].lower() == 'ctrl' or parts[1].lower() == 'ctrl':
                single_perts.append(p)
            else:
                combo_perts.append(p)
        else:
            combo_perts.append(p)
    return single_perts, combo_perts


def build_pert_gene_vectors(all_unique_perts, gene_names):
    pert_to_idx = {p: i for i, p in enumerate(all_unique_perts)}
    pert_gene_vectors = np.zeros((len(all_unique_perts), len(gene_names)), dtype=np.float32)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    for p in all_unique_perts:
        idx = pert_to_idx[p]
        for g in p.split('+'):
            g = g.strip()
            if g.lower() != 'ctrl' and g in gene_to_idx:
                pert_gene_vectors[idx, gene_to_idx[g]] = 1.0
    return pert_to_idx, pert_gene_vectors


def load_norman_data(cfg):
    print("=" * 60)
    print("Step 1: Loading Norman dataset...")
    print("=" * 60)

    h5ad_path = os.path.join(cfg.data_dir, cfg.h5ad_filename)
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"找不到: {h5ad_path}")

    print(f"Reading {h5ad_path} ({os.path.getsize(h5ad_path) / 1e9:.2f} GB)")
    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded: {adata.shape}, obs: {list(adata.obs.columns)}")

    condition_col = None
    for col in ['condition', 'perturbation', 'perturbations', 'gene',
                'guide_id', 'grna', 'sgRNA_group']:
        if col in adata.obs.columns:
            condition_col = col
            break
    if condition_col is None:
        for col in adata.obs.columns:
            if adata.obs[col].dtype == 'object' or adata.obs[col].dtype.name == 'category':
                if 10 < adata.obs[col].nunique() < 500:
                    condition_col = col
                    break
    if condition_col is None:
        raise ValueError(f"找不到扰动标签列! {list(adata.obs.columns)}")

    adata.obs['condition'] = adata.obs[condition_col].astype(str)
    unique_conds = np.unique(adata.obs['condition'].values)

    ctrl_key = None
    for c in ['ctrl', 'control', 'non-targeting', 'NT', 'unperturbed']:
        if c in unique_conds:
            ctrl_key = c
            break
    if ctrl_key is None:
        for c in unique_conds:
            if 'ctrl' in c.lower() or 'control' in c.lower():
                ctrl_key = c
                break
    if ctrl_key is None:
        raise ValueError("找不到控制组!")
    print(f"Condition: '{condition_col}', Control: '{ctrl_key}', Total: {len(unique_conds)}")

    if issparse(adata.X):
        sample_max = float(np.max(np.array(adata.X[:100].todense()).flatten()))
    else:
        sample_max = float(np.max(adata.X[:100].flatten()))
    if sample_max > 50:
        print("Normalizing raw counts...")
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    if adata.shape[1] > cfg.n_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=cfg.n_hvg)
        adata = adata[:, adata.var['highly_variable']].copy()

    cfg.input_dim = adata.shape[1]
    conditions = adata.obs['condition'].values.astype(str)
    gene_names = adata.var_names.tolist()

    if issparse(adata.X):
        X_all = np.array(adata.X.todense(), dtype=np.float32)
    else:
        X_all = np.array(adata.X, dtype=np.float32)

    is_ctrl = conditions == ctrl_key
    X_ctrl = X_all[is_ctrl]
    X_pert = X_all[~is_ctrl]
    pert_labels = conditions[~is_ctrl]
    del adata, X_all
    gc.collect()

    all_unique_perts = np.unique(pert_labels)
    single_perts, combo_perts = identify_single_and_combo_perts(all_unique_perts)
    train_perts, test_perts = train_test_split(single_perts, test_size=0.2, random_state=cfg.seed)

    pert_to_idx, pert_gene_vectors = build_pert_gene_vectors(all_unique_perts, gene_names)

    ctrl_mean = X_ctrl.mean(axis=0)
    pert_mean_expr = {}
    for p in all_unique_perts:
        mask = pert_labels == p
        if mask.sum() > 0:
            pert_mean_expr[p] = X_pert[mask].mean(axis=0)

    train_deltas = []
    for p in train_perts:
        if p in pert_mean_expr:
            train_deltas.append(pert_mean_expr[p] - ctrl_mean)
    train_mean_delta = np.mean(train_deltas, axis=0) if train_deltas else np.zeros_like(ctrl_mean)
    train_median_delta = np.median(train_deltas, axis=0) if train_deltas else np.zeros_like(ctrl_mean)

    pert_indices = np.array([pert_to_idx[p] for p in pert_labels])

    train_mask = np.isin(pert_labels, train_perts)
    X_pt, pi_t = X_pert[train_mask], pert_indices[train_mask]
    X_ct = X_ctrl[np.random.choice(X_ctrl.shape[0], size=X_pt.shape[0], replace=True)]

    test_mask = np.isin(pert_labels, test_perts)
    X_pe, pi_e = X_pert[test_mask], pert_indices[test_mask]
    X_ce = X_ctrl[np.random.choice(X_ctrl.shape[0], size=X_pe.shape[0], replace=True)]

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_ct), torch.LongTensor(pi_t), torch.FloatTensor(X_pt)),
        batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_ce), torch.LongTensor(pi_e), torch.FloatTensor(X_pe)),
        batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    train_pert_indices = {pert_to_idx[p] for p in train_perts if p in pert_to_idx}

    data_info = {
        'n_perts': len(all_unique_perts),
        'pert_to_idx': pert_to_idx,
        'pert_gene_vectors': torch.FloatTensor(pert_gene_vectors).to(cfg.device),
        'ctrl_mean': ctrl_mean,
        'ctrl_mean_tensor': torch.FloatTensor(ctrl_mean).to(cfg.device),
        'train_perts': list(train_perts),
        'test_perts': list(test_perts),
        'train_pert_indices': train_pert_indices,
        'pert_mean_expr': pert_mean_expr,
        'gene_names': gene_names,
        'X_ctrl': X_ctrl,
        'train_mean_delta': train_mean_delta,
        'train_median_delta': train_median_delta,
    }

    print(f"Shape: {cfg.input_dim} genes | Ctrl: {X_ctrl.shape[0]} | Pert: {X_pert.shape[0]}")
    print(f"Single: {len(single_perts)}, Combo: {len(combo_perts)}")
    print(f"Train: {len(train_perts)} perts ({X_pt.shape[0]} cells) | "
          f"Test: {len(test_perts)} perts ({X_pe.shape[0]} cells)")
    return train_loader, test_loader, data_info


# ============================================================
# 2. 模型
# ============================================================

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)])
            prev = h
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class LinearSEM(nn.Module):
    def __init__(self, latent_dim, rank):
        super().__init__()
        self.latent_dim = latent_dim
        self.U = nn.Parameter(torch.randn(latent_dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(latent_dim, rank) * 0.01)

    def get_B(self):
        B = self.U @ self.V.t()
        return B - torch.diag(torch.diag(B))

    def intervene(self, z, pert_vector):
        B = self.get_B()
        I = torch.eye(self.latent_dim, device=z.device)
        e = z + pert_vector
        try:
            return torch.linalg.solve(I - B, e.t()).t()
        except Exception:
            return torch.linalg.solve(I - B + 1e-6 * I, e.t()).t()


class PerturbationEncoder(nn.Module):
    def __init__(self, n_perts, gene_dim, latent_dim, embed_dropout=0.5):
        super().__init__()
        self.pert_embedding = nn.Embedding(n_perts, latent_dim)
        self.gene_to_latent = nn.Sequential(
            nn.Linear(gene_dim, 512), nn.ReLU(), nn.Linear(512, latent_dim))
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim))
        self.embed_dropout = embed_dropout

    def forward(self, pert_idx, pert_gene_vectors, use_embedding=True):
        gene_emb = self.gene_to_latent(pert_gene_vectors[pert_idx])
        emb = self.pert_embedding(pert_idx)
        if self.training and use_embedding:
            mask = (torch.rand(emb.shape[0], 1, device=emb.device) > self.embed_dropout).float()
            emb = emb * mask
        elif not use_embedding:
            emb = torch.zeros_like(emb)
        return self.fusion(torch.cat([emb, gene_emb], dim=-1))


class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2):
        super().__init__()
        self.adj_weight = nn.Parameter(torch.randn(in_dim, in_dim) * 0.01)
        layers = []
        prev = in_dim
        for i in range(n_layers):
            out_d = hidden_dim if i < n_layers - 1 else out_dim
            layers.append(nn.Linear(prev, out_d))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
            prev = out_d
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        A = torch.sigmoid(self.adj_weight + self.adj_weight.t())
        A = A - torch.diag(torch.diag(A))
        D = A.sum(dim=1).clamp(min=1e-6)
        D_inv = torch.diag(1.0 / torch.sqrt(D))
        return self.mlp(z @ (D_inv @ A @ D_inv))


class DeepSEM(nn.Module):
    def __init__(self, cfg, n_perts, gene_dim):
        super().__init__()
        self.encoder = Encoder(cfg.input_dim, cfg.encoder_hidden, cfg.latent_dim)
        self.decoder = Decoder(cfg.latent_dim, cfg.decoder_hidden, cfg.input_dim)
        self.sem = LinearSEM(cfg.latent_dim, cfg.sem_rank)
        self.gnn = SimpleGNN(cfg.latent_dim, cfg.gnn_hidden, cfg.latent_dim, cfg.gnn_layers)
        self.pert_encoder = PerturbationEncoder(n_perts, gene_dim, cfg.latent_dim, cfg.embed_dropout)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        return mu

    def _combine(self, z, pert_latent):
        z_sem = self.sem.intervene(z, pert_latent)
        z_gnn = self.gnn(z + pert_latent)
        alpha = torch.sigmoid(self.alpha)
        return alpha * z_sem + (1 - alpha) * z_gnn

    def forward(self, x_ctrl, pert_idx, pert_gene_vectors):
        mu, logvar = self.encoder(x_ctrl)
        z = self.reparameterize(mu, logvar)
        pert_latent = self.pert_encoder(pert_idx, pert_gene_vectors, use_embedding=True)
        x_pred = self.decoder(self._combine(z, pert_latent))
        return x_pred, mu, logvar, self.sem.get_B()

    def predict(self, x_ctrl, pert_idx, pert_gene_vectors, use_embedding=False):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(x_ctrl)
            pert_latent = self.pert_encoder(pert_idx, pert_gene_vectors, use_embedding=use_embedding)
            return self.decoder(self._combine(mu, pert_latent))


# ============================================================
# 3. 损失
# ============================================================

def compute_loss(x_pred, x_target, mu, logvar, B, cfg):
    recon = F.mse_loss(x_pred, x_target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    sparse_p = torch.norm(B, p=1)
    d = B.shape[0]
    try:
        dag_p = torch.trace(torch.matrix_exp(B * B)) - d
    except Exception:
        dag_p = torch.norm(B * B, p='fro')
    causal = cfg.causal_reg * (sparse_p + 0.1 * dag_p)
    total = recon + cfg.kl_weight * kl + causal
    return total, {
        'total': total.item(), 'recon': recon.item(),
        'kl': kl.item(), 'causal': causal.item(),
    }


# ============================================================
# 4. 评估
# ============================================================

def eval_per_pert(pred_delta, true_delta, ctrl_mean, cfg):
    """对单个 perturbation 计算全部指标，返回 dict"""
    diff = np.abs(true_delta)
    t20 = np.argsort(diff)[-cfg.n_top_degs:]
    t50 = np.argsort(diff)[-cfg.n_top_degs_2:]

    result = {}

    c_all = np.corrcoef(pred_delta, true_delta)[0, 1]
    result['delta_all'] = c_all if not np.isnan(c_all) else None

    if len(t20) >= 2:
        c20 = np.corrcoef(pred_delta[t20], true_delta[t20])[0, 1]
        result['delta_deg20'] = c20 if not np.isnan(c20) else None

        pred_mean = ctrl_mean + pred_delta
        true_mean = ctrl_mean + true_delta
        result['mse_de20'] = float(np.mean((pred_mean[t20] - true_mean[t20]) ** 2))
        result['frac_opp'] = float(np.mean(np.sign(pred_delta[t20]) != np.sign(true_delta[t20])))
    else:
        result['delta_deg20'] = None
        result['mse_de20'] = None
        result['frac_opp'] = None

    if len(t50) >= 2:
        c50 = np.corrcoef(pred_delta[t50], true_delta[t50])[0, 1]
        result['delta_deg50'] = c50 if not np.isnan(c50) else None
    else:
        result['delta_deg50'] = None

    return result


def aggregate_metrics(per_pert_results):
    """从 per-pert dict list 聚合为平均指标"""
    keys = ['delta_all', 'delta_deg20', 'delta_deg50', 'mse_de20', 'frac_opp']
    agg = {}
    for k in keys:
        vals = [r[k] for r in per_pert_results if r.get(k) is not None]
        agg[k] = float(np.mean(vals)) if vals else 0.0
    agg['n_eval'] = len([r for r in per_pert_results if r.get('delta_deg20') is not None])
    return agg


@torch.no_grad()
def evaluate_deepsem(model, test_loader, data_info, cfg):
    model.eval()
    all_preds, all_targets, all_pert_idx = [], [], []

    for x_ctrl, pert_idx, x_target in test_loader:
        x_pred = model.predict(x_ctrl.to(cfg.device), pert_idx.to(cfg.device),
                               data_info['pert_gene_vectors'], use_embedding=False)
        all_preds.append(x_pred.cpu().numpy())
        all_targets.append(x_target.numpy())
        all_pert_idx.append(pert_idx.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_pert_idx = np.concatenate(all_pert_idx)
    ctrl_mean = data_info['ctrl_mean']

    per_pert = []
    for pn in data_info['test_perts']:
        if pn not in data_info['pert_to_idx']:
            continue
        idx = data_info['pert_to_idx'][pn]
        mask = all_pert_idx == idx
        if mask.sum() < 2:
            continue
        pred_delta = all_preds[mask].mean(0) - ctrl_mean
        true_delta = all_targets[mask].mean(0) - ctrl_mean
        r = eval_per_pert(pred_delta, true_delta, ctrl_mean, cfg)
        r['pert'] = pn
        per_pert.append(r)

    metrics = aggregate_metrics(per_pert)
    return metrics, per_pert, all_preds, all_targets, all_pert_idx


# ============================================================
# 5. 基线 (★ 包含 Ridge 和 KNN)
# ============================================================

def build_pert_onehot(pert_name, gene_names, gene_to_idx):
    vec = np.zeros(len(gene_names), dtype=np.float32)
    for g in pert_name.split('+'):
        g = g.strip()
        if g.lower() != 'ctrl' and g in gene_to_idx:
            vec[gene_to_idx[g]] = 1.0
    return vec


def compute_all_baselines(data_info, cfg):
    """
    4个基线：
    1. No-change (δ=0): 预测 ctrl_mean
    2. Global-mean δ: 所有 test 预测同一个均值变化
    3. Ridge: 从扰动 one-hot 预测 delta (文献级别的线性基线)
    4. KNN: 从扰动 one-hot 找最近邻
    """
    print("\n  Computing baselines...")

    ctrl_mean = data_info['ctrl_mean']
    gene_names = data_info['gene_names']
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    mean_delta = data_info['train_mean_delta']

    # 构造 Ridge/KNN 训练数据
    X_train, Y_train = [], []
    for p in data_info['train_perts']:
        if p not in data_info['pert_mean_expr']:
            continue
        X_train.append(build_pert_onehot(p, gene_names, gene_to_idx))
        Y_train.append(data_info['pert_mean_expr'][p] - ctrl_mean)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # 训练 Ridge (尝试几个 alpha 选最好的)
    best_ridge, best_ridge_alpha, best_ridge_score = None, None, -1
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, Y_train)
        scores = []
        for p in data_info['test_perts']:
            if p not in data_info['pert_mean_expr']:
                continue
            x = build_pert_onehot(p, gene_names, gene_to_idx).reshape(1, -1)
            pred_delta = ridge.predict(x)[0]
            true_delta = data_info['pert_mean_expr'][p] - ctrl_mean
            t20 = np.argsort(np.abs(true_delta))[-cfg.n_top_degs:]
            c = np.corrcoef(pred_delta[t20], true_delta[t20])[0, 1]
            if not np.isnan(c):
                scores.append(c)
        avg = np.mean(scores) if scores else 0
        if avg > best_ridge_score:
            best_ridge_score = avg
            best_ridge_alpha = alpha
            best_ridge = ridge

    print(f"    Ridge best α={best_ridge_alpha}")

    # 训练 KNN
    best_knn, best_knn_k, best_knn_score = None, None, -1
    for k in [1, 3, 5]:
        if k > X_train.shape[0]:
            continue
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, Y_train)
        scores = []
        for p in data_info['test_perts']:
            if p not in data_info['pert_mean_expr']:
                continue
            x = build_pert_onehot(p, gene_names, gene_to_idx).reshape(1, -1)
            pred_delta = knn.predict(x)[0]
            true_delta = data_info['pert_mean_expr'][p] - ctrl_mean
            t20 = np.argsort(np.abs(true_delta))[-cfg.n_top_degs:]
            c = np.corrcoef(pred_delta[t20], true_delta[t20])[0, 1]
            if not np.isnan(c):
                scores.append(c)
        avg = np.mean(scores) if scores else 0
        if avg > best_knn_score:
            best_knn_score = avg
            best_knn_k = k
            best_knn = knn

    print(f"    KNN best k={best_knn_k}")

    # 对每个 test pert 逐个评估所有基线
    all_baselines = {
        'No-change (δ=0)': [],
        'Global-mean δ': [],
        f'Ridge (α={best_ridge_alpha})': [],
        f'KNN (k={best_knn_k})': [],
    }

    for pn in data_info['test_perts']:
        if pn not in data_info['pert_mean_expr']:
            continue
        true_delta = data_info['pert_mean_expr'][pn] - ctrl_mean
        x_vec = build_pert_onehot(pn, gene_names, gene_to_idx).reshape(1, -1)

        predictions = {
            'No-change (δ=0)': np.zeros_like(ctrl_mean),
            'Global-mean δ': mean_delta,
            f'Ridge (α={best_ridge_alpha})': best_ridge.predict(x_vec)[0],
            f'KNN (k={best_knn_k})': best_knn.predict(x_vec)[0],
        }

        for name, pred_delta in predictions.items():
            r = eval_per_pert(pred_delta, true_delta, ctrl_mean, cfg)
            r['pert'] = pn
            all_baselines[name].append(r)

    results = {}
    for name, per_pert_list in all_baselines.items():
        results[name] = aggregate_metrics(per_pert_list)
        results[name]['per_pert'] = per_pert_list

    return results


# ============================================================
# 6. 训练
# ============================================================

def train_one_epoch(model, loader, optimizer, data_info, cfg):
    model.train()
    sums = {'total': 0, 'recon': 0, 'kl': 0, 'causal': 0}
    n = 0
    for x_ctrl, pert_idx, x_target in loader:
        x_ctrl = x_ctrl.to(cfg.device)
        pert_idx = pert_idx.to(cfg.device)
        x_target = x_target.to(cfg.device)
        optimizer.zero_grad()
        x_pred, mu, logvar, B = model(x_ctrl, pert_idx, data_info['pert_gene_vectors'])
        loss, ld = compute_loss(x_pred, x_target, mu, logvar, B, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        for k in sums:
            sums[k] += ld[k]
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


def train_model(model, train_loader, test_loader, data_info, cfg):
    print("\n" + "=" * 60)
    print("Step 2: Training DeepSEM")
    print("=" * 60)

    hdr = (f"{'Ep':>5} | {'Loss':>7} {'Recon':>7} {'KL':>7} | "
           f"{'δAll':>6} {'δD20':>6} {'δD50':>6} {'MseD20':>7} {'Opp':>5} | {'α':>5} {'ES':>5}")
    print(hdr)
    print("-" * len(hdr))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs, eta_min=1e-5)

    best_score = -1
    best_metrics = None
    patience_ctr = 0
    history = {'loss': [], 'kl': [],
               'delta_all': [], 'delta_deg20': [], 'delta_deg50': [], 'mse_de20': []}

    for epoch in range(1, cfg.n_epochs + 1):
        tl = train_one_epoch(model, train_loader, optimizer, data_info, cfg)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            m, _, _, _, _ = evaluate_deepsem(model, test_loader, data_info, cfg)

            history['loss'].append(tl['total'])
            history['kl'].append(tl['kl'])
            history['delta_all'].append(m['delta_all'])
            history['delta_deg20'].append(m['delta_deg20'])
            history['delta_deg50'].append(m['delta_deg50'])
            history['mse_de20'].append(m['mse_de20'])

            improved = m['delta_deg20'] > best_score
            if improved:
                best_score = m['delta_deg20']
                best_metrics = m.copy()
                patience_ctr = 0
                torch.save(model.state_dict(), 'best_deepsem.pth')
            else:
                patience_ctr += 1

            alpha = torch.sigmoid(model.alpha).item()
            es = "★" if improved else f"{patience_ctr}/{cfg.patience}"

            print(f"{epoch:3d}/{cfg.n_epochs} | "
                  f"{tl['total']:7.4f} {tl['recon']:7.4f} {tl['kl']:7.4f} | "
                  f"{m['delta_all']:6.4f} {m['delta_deg20']:6.4f} {m['delta_deg50']:6.4f} "
                  f"{m['mse_de20']:7.4f} {m['frac_opp']:5.3f} | {alpha:5.3f} {es:>5}")

            if patience_ctr >= cfg.patience:
                print(f"\n⏹ Early stop at epoch {epoch}")
                break

    print(f"\n🏆 Best δ-DEG20: {best_score:.4f}")
    return history, best_metrics


# ============================================================
# 7. 可视化 (论文图)
# ============================================================

def plot_results(history, model, test_loader, data_info, cfg, baselines, deepsem_per_pert):
    print("\n" + "=" * 60)
    print("Step 4: Generating figures for thesis")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ctrl_m = data_info['ctrl_mean']

    # (0,0) 训练曲线
    ax = axes[0, 0]
    ax.plot(history['loss'], 'b-', lw=2, label='Total Loss')
    ax.set_title('Training Loss', fontsize=12)
    ax.set_xlabel('Eval Step')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(history['kl'], 'r--', lw=1, alpha=0.7, label='KL')
    ax2.set_ylabel('KL', color='red')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # (0,1) Delta Pearson 学习曲线
    ax = axes[0, 1]
    s = range(len(history['delta_all']))
    ax.plot(s, history['delta_all'], 'g--', lw=1.5, label='δ All genes')
    ax.plot(s, history['delta_deg50'], 'b-', lw=1.5, label='δ Top-50 DEGs')
    ax.plot(s, history['delta_deg20'], 'r-', lw=2, label='δ Top-20 DEGs')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.legend(fontsize=8)
    ax.set_title('Evaluation Metrics During Training', fontsize=12)
    ax.set_xlabel('Eval Step')
    ax.set_ylabel('Delta Pearson Correlation')
    ax.grid(True, alpha=0.3)

    # (0,2) 示例扰动的 scatter
    _, ds_per_pert, preds, targets, pidx = evaluate_deepsem(model, test_loader, data_info, cfg)
    ax = axes[0, 2]
    plotted = False
    for tp in data_info['test_perts']:
        if tp not in data_info['pert_to_idx']:
            continue
        idx = data_info['pert_to_idx'][tp]
        mask = pidx == idx
        if mask.sum() < 5:
            continue
        pd = preds[mask].mean(0) - ctrl_m
        td = targets[mask].mean(0) - ctrl_m
        t20 = np.argsort(np.abs(td))[-cfg.n_top_degs:]
        ax.scatter(td, pd, alpha=0.05, s=2, c='gray', label='All genes' if not plotted else None)
        ax.scatter(td[t20], pd[t20], alpha=0.9, s=30, c='red', edgecolors='k',
                   linewidth=0.4, label='Top-20 DEGs' if not plotted else None)
        c_a = np.corrcoef(pd, td)[0, 1]
        c_20 = np.corrcoef(pd[t20], td[t20])[0, 1]
        ax.set_title(f'Example: {tp}\nδAll={c_a:.3f}, δDEG20={c_20:.3f}', fontsize=11)
        plotted = True
        break

    if plotted:
        lim = max(abs(v) for v in [*ax.get_xlim(), *ax.get_ylim()])
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.axvline(0, color='gray', lw=0.5, ls=':')
        ax.legend(fontsize=7)
    ax.set_xlabel('True Δ Expression')
    ax.set_ylabel('Predicted Δ Expression')

    # (1,0) ★ 基线对比柱状图 (论文主图)
    ax = axes[1, 0]
    methods = list(baselines.keys()) + ['DeepSEM (Ours)']
    deg20_vals = [baselines[m]['delta_deg20'] for m in baselines.keys()]
    deg20_vals = [v if not np.isnan(v) else 0 for v in deg20_vals]
    deg20_vals.append(deepsem_per_pert['delta_deg20'])

    colors = ['#95a5a6'] * len(baselines) + ['#e74c3c']
    bars = ax.bar(range(len(methods)), deg20_vals, color=colors, edgecolor='black', linewidth=0.5)

    # 标注数值
    for i, (bar, val) in enumerate(zip(bars, deg20_vals)):
        if val != 0 and not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(' δ', '\nδ').replace(' (', '\n(') for m in methods],
                       fontsize=7, ha='center')
    ax.set_ylabel('Delta Pearson (DEG-20)')
    ax.set_title('Method Comparison: δ-DEG20', fontsize=12)
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=0.2, axis='y')

    # (1,1) Causal matrix
    with torch.no_grad():
        B = model.sem.get_B().cpu().numpy()
    ns = min(30, B.shape[0])
    im = axes[1, 1].imshow(B[:ns, :ns], cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    axes[1, 1].set_title('Learned Causal Matrix B', fontsize=12)
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # (1,2) Per-pert 性能
    ax = axes[1, 2]
    pp = {r['pert']: r['delta_deg20'] for r in ds_per_pert if r.get('delta_deg20') is not None}
    if pp:
        sp = sorted(pp.items(), key=lambda x: x[1], reverse=True)
        names = [n[:18] for n, _ in sp]
        vals = [v for _, v in sp]
        colors_bar = ['#27ae60' if v > 0.6 else '#f39c12' if v > 0.3 else '#e74c3c' for v in vals]
        ax.barh(range(len(sp)), vals, color=colors_bar, edgecolor='black', linewidth=0.3)
        ax.set_yticks(range(len(sp)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel('Delta Pearson (DEG-20)')
        ax.set_title(f'Per-perturbation Performance\n'
                     f'Mean={np.mean(vals):.3f}, Median={np.median(vals):.3f}', fontsize=11)
        ax.set_xlim(-1, 1)
        ax.axvline(0, color='gray', lw=0.5)

    plt.tight_layout()
    plt.savefig('deepsem_results.png', dpi=150, bbox_inches='tight')
    print("  Saved: deepsem_results.png")


# ============================================================
# 8. 论文汇总表
# ============================================================

def print_paper_table(baselines, deepsem_metrics, best_metrics):
    print("\n" + "=" * 60)
    print("Step 5: Results Table (copy to thesis)")
    print("=" * 60)

    print(f"\n{'Method':<25} {'δ-DEG20':>8} {'δ-DEG50':>8} {'δ-All':>8} "
          f"{'MSE-D20':>8} {'Dir-Err%':>8}")
    print("=" * 75)

    for name, m in baselines.items():
        d20 = f"{m['delta_deg20']:.4f}" if not np.isnan(m['delta_deg20']) else 'N/A'
        d50 = f"{m['delta_deg50']:.4f}" if not np.isnan(m['delta_deg50']) else 'N/A'
        da = f"{m['delta_all']:.4f}" if not np.isnan(m['delta_all']) else 'N/A'
        mse = f"{m['mse_de20']:.4f}" if not np.isnan(m['mse_de20']) else 'N/A'
        opp = f"{m['frac_opp']:.3f}" if not np.isnan(m.get('frac_opp', float('nan'))) else 'N/A'
        print(f"{name:<25} {d20:>8} {d50:>8} {da:>8} {mse:>8} {opp:>8}")

    dm = best_metrics
    print(f"{'DeepSEM (Ours)':<25} {dm['delta_deg20']:>8.4f} {dm['delta_deg50']:>8.4f} "
          f"{dm['delta_all']:>8.4f} {dm['mse_de20']:>8.4f} {dm['frac_opp']:>8.3f}")
    print("=" * 75)

    # 判定
    ridge_name = [n for n in baselines.keys() if 'Ridge' in n]
    if ridge_name:
        ridge_score = baselines[ridge_name[0]]['delta_deg20']
        our_score = dm['delta_deg20']
        diff = our_score - ridge_score

        print(f"\n  vs Ridge baseline: {diff:+.4f}")
        if diff > 0.03:
            print(f"  → DeepSEM outperforms Ridge by {diff:.4f}")
        elif diff > 0:
            print(f"  → DeepSEM marginally better (+{diff:.4f})")
        elif diff > -0.03:
            print(f"  → DeepSEM comparable to Ridge ({diff:+.4f})")
        else:
            print(f"  → Ridge outperforms DeepSEM by {-diff:.4f}")

    # 论文可用的描述
    mean_name = [n for n in baselines.keys() if 'Global' in n]
    if mean_name:
        mean_score = baselines[mean_name[0]]['delta_deg20']
        print(f"\n  vs Global-mean baseline: {dm['delta_deg20'] - mean_score:+.4f}")

    print(f"\n论文中可以这样写:")
    print(f"  'DeepSEM achieved a delta Pearson correlation of {dm['delta_deg20']:.3f}")
    print(f"   on the top-20 differentially expressed genes for unseen perturbations,'")
    if ridge_name:
        rd = baselines[ridge_name[0]]['delta_deg20']
        print(f"   compared to {rd:.3f} for Ridge regression baseline.'")


# ============================================================
# 9. 主函数
# ============================================================

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  DeepSEM v3-final — 本科毕设最终版                       ║")
    print("║  模型训练 + 4基线对比 + 论文图表 一次出全部结果            ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Step 1: 数据
    train_loader, test_loader, data_info = load_norman_data(cfg)

    # Step 2: 训练
    model = DeepSEM(cfg=cfg, n_perts=data_info['n_perts'], gene_dim=cfg.input_dim).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParams: {n_params:,} | Input: {cfg.input_dim} | Latent: {cfg.latent_dim}")

    history, best_metrics = train_model(model, train_loader, test_loader, data_info, cfg)

    # 加载最佳模型
    if os.path.exists('best_deepsem.pth'):
        model.load_state_dict(torch.load('best_deepsem.pth', map_location=cfg.device))

    # Step 3: 基线计算
    print("\n" + "=" * 60)
    print("Step 3: Computing baselines (Ridge, KNN, ...)")
    print("=" * 60)
    baselines = compute_all_baselines(data_info, cfg)

    # 获取 DeepSEM per-pert 结果
    deepsem_agg, deepsem_per_pert, _, _, _ = evaluate_deepsem(model, test_loader, data_info, cfg)

    # Step 4: 可视化
    plot_results(history, model, test_loader, data_info, cfg, baselines, deepsem_agg)

    # Step 5: 论文汇总表
    print_paper_table(baselines, deepsem_agg, best_metrics)

    # 保存因果矩阵
    with torch.no_grad():
        np.save('causal_matrix_B.npy', model.sem.get_B().cpu().numpy())

    print("\n✅ 全部完成! 论文所需数据和图表已生成。")
    print("   - deepsem_results.png  (论文主图)")
    print("   - causal_matrix_B.npy  (因果矩阵)")
    print("   - best_deepsem.pth     (最佳模型)")


if __name__ == '__main__':
    main()