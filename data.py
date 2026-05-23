import scanpy as sc
import pandas as pd

path = r"C:\Users\hbr\PythonProject26\data\adamson\perturb_processed.h5ad"
adata = sc.read_h5ad(path)

print("shape:", adata.shape)
print("obs columns:", list(adata.obs.columns))
print("var columns:", list(adata.var.columns))

# 你论文里说 condition 字段，优先检查
col = "condition"
if col not in adata.obs.columns:
    print("condition 不存在，请从 obs columns 里确认扰动标签列")
else:
    vc = adata.obs[col].value_counts()
    print("\nTop conditions:")
    print(vc.head(20))

    ctrl_mask = adata.obs[col].astype(str).str.lower().isin(["ctrl", "control", "non-targeting"])
    print("\nctrl cells:", int(ctrl_mask.sum()))
    print("pert cells:", int((~ctrl_mask).sum()))
    print("total conditions:", adata.obs[col].nunique())

    conditions = [str(x) for x in adata.obs[col].unique()]
    non_ctrl = [x for x in conditions if x.lower() not in ["ctrl", "control", "non-targeting"]]

    single = []
    combo = []
    for c in non_ctrl:
        parts = [p for p in c.split("+") if p.lower() != "ctrl"]
        if len(parts) == 1:
            single.append(c)
        elif len(parts) >= 2:
            combo.append(c)

    print("single perturbations:", len(single))
    print("combo perturbations:", len(combo))
    print("total non-ctrl perturbations:", len(non_ctrl))