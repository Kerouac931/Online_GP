"""
VFE GP Runner — Joint Optimization Version
===========================================
基于 Titsias (2009) VFE Sparse GP，遵循原论文做法：
单阶段联合优化 Xm 和超参数 (lengthscale, signal_var, noise_var)。

训练模式:
  'train_only'      — 只用 train 数据 fit()，validate 评估
  'train_and_online'— train + online 合并为 full-batch，fit()，validate 评估

优化策略：
  单阶段 Adam，同时优化 Xm + 超参数，对应 Titsias (2009) 中
  "jointly maximize F_V w.r.t. hyperparameters and inducing inputs"。
  与原论文差异：使用 Adam 替代共轭梯度法（conjugate gradients）。

训练保存策略：
  记录 F_V 最高时的参数，训练结束后恢复最佳参数再评估。
"""
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import pandas as pd




# ================================================================
# ===== 数据集类（直接复用，不改动）==============================
# ================================================================
class FunctionBM:
    def __init__(self, name):
        self.paths = {
            'himmelblau':   r'd:\project\project_GP\code\MA_GP-main\data_lu\four_func_2500\Himmelblau.csv',
            'rastrigin':    r'd:\project\project_GP\code\MA_GP-main\data_lu\four_func_2500\Rastrigin.csv',
            'rosenbrock':   r'd:\project\project_GP\code\MA_GP-main\data_lu\four_func_2500\Rosenbrock.csv',
            'sixhumpcamel': r'd:\project\project_GP\code\MA_GP-main\data_lu\four_func_2500\SixHumpCamel.csv',
        }
        self.name = name.lower()
        if self.name not in self.paths:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(self.paths.keys())}")
        self.path = self.paths[self.name]
        self.split_config = {'train_end': 100, 'online_end': 600}

    def load_data(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"路径未找到: {self.path}")
        data = pd.read_csv(self.path)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X, y

    def get_splits(self):
        X, y = self.load_data()
        t, o = self.split_config['train_end'], self.split_config['online_end']
        return {
            'train':    {'X': X[:t],  'y': y[:t]},
            'online':   {'X': X[t:o], 'y': y[t:o]},
            'validate': {'X': X[o:],  'y': y[o:]},
        }


class StaticDatasetBM:
    def __init__(self, name):
        paths = {
            'boston':   r'd:\project\project_GP\code\MA_GP-main\data_lu\boston_concrete\BostonHousing_randomized.csv',
            'concrete': r'd:\project\project_GP\code\MA_GP-main\data_lu\boston_concrete\Concrete.csv',
        }
        self.name = name.lower()
        if self.name not in paths:
            raise ValueError(f"Unknown static dataset: {name}")
        self.path = paths[self.name]
        self.split_config = {
            'boston':   {'train_end': 100, 'online_end': 451},
            'concrete': {'train_end': 100, 'online_end': 927},
        }

    def load_data(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"路径未找到: {self.path}")
        data = pd.read_csv(self.path)
        return data.iloc[:, :-1].values.astype(float), data.iloc[:, -1].values.astype(float)

    def get_splits(self, online_end=None):
        X, y = self.load_data()
        cfg = self.split_config[self.name]
        t = cfg['train_end']
        o = online_end if online_end is not None else cfg['online_end']
        return {
            'train':    {'X': X[:t],  'y': y[:t]},
            'online':   {'X': X[t:o], 'y': y[t:o]},
            'validate': {'X': X[o:],  'y': y[o:]},
        }


class SysIDBM:
    def __init__(self, name):
        self.base_dir = r'd:\project\project_GP\code\MA_GP-main\data_lu\sysID'
        self.name = name.lower()
        self.file_map = {
            'vanderpol': 'VanDerPol.csv',
            'building':  'Building.csv',
            'boucwen':   'BoucWen.csv',
            'tanks':     'Tanks.csv',
        }
        self.validate_file_map = {
            'boucwen': 'validate_BoucWen.csv',
            'tanks':   'validate_Tanks.csv',
        }
        if self.name not in self.file_map:
            raise ValueError(f"Unknown SysID dataset: {name}")

    def _read_csv(self, filename):
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到文件: {path}")
        index_col = 0 if self.name == 'building' else None # first column in building not included in training data
        return pd.read_csv(path, index_col=index_col)

    def get_splits(self):
        data = self._read_csv(self.file_map[self.name])
        result = {'train': data.iloc[:100]}
        if self.name == 'vanderpol':
            result['online']   = data.iloc[100:1000]
            result['validate'] = data.iloc[1000:]
        elif self.name == 'building':
            result['train']    = data.iloc[:1000]
            result['online']   = data.iloc[1000:17520]
            result['validate'] = data.iloc[17520:]
        elif self.name in ['boucwen', 'tanks']:
            result['online'] = data.iloc[100:]
            if self.name in self.validate_file_map:
                result['validate'] = self._read_csv(self.validate_file_map[self.name])
            else:
                result['validate'] = pd.DataFrame()
        return result


# ================================================================
# ===== 配置区 ====================================================
# ================================================================


# 四函数:   'rastrigin' , 'rosenbrock' , 'himmelblau' , 'sixhumpcamel'
# 静态:     'boston'    , 'concrete'
# 系统辨识: 'vanderpol' , 'boucwen', 'tanks', 'building'

#DATASET_LIST = ['rastrigin','rosenbrock','himmelblau','sixhumpcamel','boston','concrete','vanderpol','boucwen','tanks']
DATASET_LIST = ['building']  

M_LIST = [50]


# 'train_online' = initial batch
# 'train_and_online'= train + online = full batch

#'train_only', 'train_and_online' 
MODE_LIST = ['train_only', 'train_and_online']


TRAIN_CONFIG_MAP = {
    # 四函数 — 单阶段联合优化 Xm + 超参数，对应 Titsias (2009) 原文做法
    'himmelblau':   {'epochs': 500, 'lr': 0.01,  'patience': 30},
    'rastrigin':    {'epochs': 500, 'lr': 0.01,  'patience': 30},
    'rosenbrock':   {'epochs': 500, 'lr': 0.01,  'patience': 30},
    'sixhumpcamel': {'epochs': 500, 'lr': 0.01,  'patience': 30},
    # 静态数据集
    'boston':       {'epochs': 700, 'lr': 0.005, 'patience': 40},
    'concrete':     {'epochs': 700, 'lr': 0.005, 'patience': 50},
    # 系统辨识
    'vanderpol':    {'epochs': 800, 'lr': 0.005, 'patience': 50},
    'boucwen':      {'epochs': 800, 'lr': 0.005, 'patience': 50},
    'tanks':        {'epochs': 800, 'lr': 0.005, 'patience': 50},
    'building':     {'epochs': 700, 'lr': 0.005, 'patience': 50},
}

MODEL_CONFIG = {
    'jitter': 1e-6,
}

OUTPUT_CONFIG = {
    'save_dir':       r'D:\project\VFE_result',
    'save_stats':     False,
    'verbose':        True,
    'print_interval': 50,
}

FUNC_DATASETS   = {'himmelblau', 'rastrigin', 'rosenbrock', 'sixhumpcamel'}
STATIC_DATASETS = {'boston', 'concrete'}
SYSID_DATASETS  = {'vanderpol', 'boucwen', 'tanks', 'building'}


# ================================================================
# ===== VFE 模型 ==================================================
# ================================================================

class RBFKernel(nn.Module):
    def __init__(self, input_dim: int, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__()
        self.log_lengthscale = nn.Parameter(
            torch.log(torch.full((input_dim,), lengthscale, dtype=torch.float64))
        )
        self.log_variance = nn.Parameter(
            torch.tensor(np.log(variance), dtype=torch.float64)
        )

    def forward(self, x1, x2):
        ls  = torch.exp(self.log_lengthscale)
        var = torch.exp(self.log_variance)
        dist_sq = torch.cdist(x1 / ls, x2 / ls, p=2) ** 2
        return var * torch.exp(-0.5 * dist_sq)

    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale).detach().numpy()

    @property
    def signal_variance(self):
        return torch.exp(self.log_variance).item()


class VFEGP:
    def __init__(self, Xm_init, noise_var=0.1, lengthscale_init=1.0,
                 variance_init=1.0, jitter=1e-6):
        self.M, self.D  = Xm_init.shape
        self.jitter      = jitter

        self.kernel   = RBFKernel(self.D, lengthscale_init, variance_init)
        self.log_beta = nn.Parameter(
            torch.tensor(np.log(1.0 / noise_var), dtype=torch.float64)
        )
        self.Xm = nn.Parameter(torch.tensor(Xm_init, dtype=torch.float64))

        self._mu    = None
        self._Sigma = None

    def all_parameters(self):
        """所有参数：超参数 + Xm，对应 Titsias (2009) 联合优化。"""
        return list(self.kernel.parameters()) + [self.log_beta, self.Xm]

    def _Kmm(self):
        return self.kernel(self.Xm, self.Xm) + \
               torch.eye(self.M, dtype=torch.float64) * self.jitter

    def _Kmn(self, X):
        return self.kernel(self.Xm, X)

    @staticmethod
    def _chol(A):
        try:
            return torch.linalg.cholesky(A)
        except RuntimeError:
            n = A.shape[0]
            return torch.linalg.cholesky(A + torch.eye(n, dtype=torch.float64) * 1e-4)

    def compute_loss(self, X, y):
        N    = X.shape[0]
        beta = torch.exp(self.log_beta)
        Kmm      = self._Kmm()
        Kmn      = self._Kmn(X)
        Knn_diag = torch.diagonal(self.kernel(X, X))

        Lm       = self._chol(Kmm)
        V        = torch.linalg.solve_triangular(Lm, Kmn, upper=False)
        Qnn_diag = torch.sum(V ** 2, dim=0)

        trace_penalty = -0.5 * beta * torch.sum(Knn_diag - Qnn_diag)

        IpbVVt  = torch.eye(self.M, dtype=torch.float64) + beta * (V @ V.T)
        L_inner = self._chol(IpbVVt)

        log_det = (
            -N * torch.log(beta)
            + 2.0 * torch.sum(torch.log(torch.diagonal(L_inner)))
        )
        Vy   = V @ y
        tmp  = torch.cholesky_solve(Vy, L_inner)
        quad = beta * (y.T @ y).squeeze() - beta ** 2 * (Vy.T @ tmp).squeeze()

        log_lik = (
            -0.5 * N * torch.log(torch.tensor(2 * np.pi, dtype=torch.float64))
            - 0.5 * log_det
            - 0.5 * quad
        )
        fv = log_lik + trace_penalty
        return -fv, fv.item()

    @torch.no_grad()
    def _compute_posterior(self, X, y):
        beta  = torch.exp(self.log_beta)
        Kmm   = self._Kmm()
        Kmn   = self._Kmn(X)
        inner = Kmm + beta * (Kmn @ Kmn.T)
        L     = self._chol(inner)
        self._mu    = beta * Kmm @ torch.cholesky_solve(Kmn @ y, L)
        self._Sigma = torch.cholesky_solve(torch.eye(self.M, dtype=torch.float64), L)

    @torch.no_grad()
    def predict(self, X_star):
        if self._mu is None:
            raise RuntimeError("请先调用 fit()。")
        X_s     = torch.as_tensor(X_star, dtype=torch.float64)
        beta    = torch.exp(self.log_beta)
        Kmm     = self._Kmm()
        Kmm_inv = torch.linalg.inv(Kmm)
        Kms     = self._Kmn(X_s)
        Kss_d   = torch.diagonal(self.kernel(X_s, X_s))
        A       = Kmm_inv @ Kms
        mean    = (Kms.T @ Kmm_inv @ self._mu).squeeze(-1)
        var     = (
            Kss_d
            - torch.sum(A * Kms, dim=0)
            + torch.sum((self._Sigma @ Kms) * Kms, dim=0)
            + 1.0 / beta
        )
        return mean.numpy(), torch.clamp(var, min=1e-8).numpy()

    def get_hyperparameters(self):
        with torch.no_grad():
            return {
                'lengthscale': self.kernel.lengthscale,
                'signal_var':  self.kernel.signal_variance,
                'noise_var':   (1.0 / torch.exp(self.log_beta)).item(),
            }


# ================================================================
# ===== 训练入口 ==================================================
# ================================================================

def fit(model, X_train, y_train, cfg, verbose=True, print_interval=50):
    """
    单阶段联合优化：同时优化 Xm + 超参数，对应 Titsias (2009) 原文做法。
    带早停 + 最佳参数保存。返回 fv_history（每 epoch 的 F_V 值）。
    """
    X = torch.as_tensor(X_train, dtype=torch.float64)
    y = torch.as_tensor(y_train, dtype=torch.float64).reshape(-1, 1)

    fv_history = []
    patience   = cfg.get('patience', 30)
    epochs     = cfg['epochs']
    lr         = cfg['lr']

    params    = model.all_parameters()
    opt       = optim.Adam(params, lr=lr)
    sch       = optim.lr_scheduler.StepLR(
        opt, step_size=max(1, epochs // 3), gamma=0.5
    )
    best_fv    = -np.inf
    best_state = None
    wait       = 0

    if verbose:
        print("=" * 65)
        print("Joint optimization: Xm + hyperparameters  "
              "(Titsias 2009)")
        print(f"  epochs={epochs}  lr={lr}  patience={patience}")
        print("=" * 65)

    for ep in range(epochs):
        opt.zero_grad()
        loss, fv = model.compute_loss(X, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        opt.step()
        sch.step()
        fv_history.append(fv)

        if verbose and ep % print_interval == 0:
            hp   = model.get_hyperparameters()
            ls   = hp['lengthscale']
            ls_s = f"{ls.round(4)}" if ls.size > 1 else f"{ls[0]:.4f}"
            print(f"  [{ep:4d}/{epochs}] F_V={fv:10.3f} | "
                  f"ℓ={ls_s}, σ²={hp['signal_var']:.4f}, "
                  f"noise={hp['noise_var']:.5f}")

        if fv > best_fv:
            best_fv    = fv
            best_state = {p: p.data.clone() for p in params}
            wait       = 0
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                if verbose:
                    print(f"  Early stop at epoch {ep}  "
                          f"(best F_V={best_fv:.3f})")
                break

    # 恢复最佳参数
    if best_state is not None:
        for p, val in best_state.items():
            p.data = val

    if verbose:
        print("\nComputing optimal q*(f_m) analytically...")
    model._compute_posterior(X, y)

    if verbose:
        print(f"Done.  Final F_V = {fv_history[-1]:.3f}")
        for k, v in model.get_hyperparameters().items():
            print(f"  {k:15s}: {v}")
        print("=" * 65)

    return fv_history


# ================================================================
# ===== 评估指标 ==================================================
# ================================================================

def compute_metrics(mean, var, y_true):
    err2     = (mean - y_true) ** 2
    smse     = float(err2.mean() / np.var(y_true))
    nlp      = float(np.mean(0.5 * err2 / var + 0.5 * np.log(2 * np.pi * var)))
    std      = np.sqrt(var)
    coverage = float(((y_true >= mean - 1.96 * std) &
                      (y_true <= mean + 1.96 * std)).mean())
    return {'smse': smse, 'nlp': nlp, 'coverage': coverage}


# ================================================================
# ===== 数据加载 ==================================================
# ================================================================

def to_Xy(split_data):
    if isinstance(split_data, pd.DataFrame):
        if split_data.empty:
            return np.empty((0, 0)), np.empty(0)
        return (split_data.iloc[:, :-1].values.astype(float),
                split_data.iloc[:, -1].values.astype(float))
    return (np.array(split_data['X'], dtype=float),
            np.array(split_data['y'], dtype=float).ravel())


def load_dataset(name):
    name = name.lower()
    if   name in FUNC_DATASETS:   splits = FunctionBM(name).get_splits()
    elif name in STATIC_DATASETS: splits = StaticDatasetBM(name).get_splits()
    elif name in SYSID_DATASETS:  splits = SysIDBM(name).get_splits()
    else: raise ValueError(f"未知数据集: {name}")

    X_tr,  y_tr  = to_Xy(splits['train'])
    X_on,  y_on  = to_Xy(splits['online'])
    X_val, y_val = to_Xy(splits['validate'])

    info = {
        'name':     name,
        'D':        X_tr.shape[1],
        'N_train':  len(X_tr),
        'N_online': len(X_on),
        'N_test':   len(X_val),
    }
    return X_tr, y_tr, X_on, y_on, X_val, y_val, info


# ================================================================
# ===== 标准化 ====================================================
# ================================================================

def build_scalers(X_train, y_train, dataset):
    col_std  = X_train.std(axis=0)
    col_mean = X_train.mean(axis=0)
    
    #if dataset.lower() == 'building':
    if True:
        # building 量纲特殊，强制全列标准化
        X_mean = col_mean
        X_std  = np.where(col_std > 1e-8, col_std + 1e-8, 1.0)
        Y_mean = float(y_train.mean())
        Y_std  = float(y_train.std()) + 1e-8
    else:
        needs_X = col_std > 2.0
        X_mean  = np.where(needs_X, col_mean, 0.0)
        X_std   = np.where(needs_X, col_std + 1e-8, 1.0)
        y_std_v = float(y_train.std())
        if y_std_v > 2.0:
            Y_mean, Y_std = float(y_train.mean()), y_std_v + 1e-8
        else:
            Y_mean, Y_std = 0.0, 1.0

    return X_mean, X_std, Y_mean, Y_std


def scale_X(X, X_mean, X_std):      return (X - X_mean) / X_std
def scale_y(y, Y_mean, Y_std):      return (y - Y_mean) / Y_std
def unscale_mean(m, Y_mean, Y_std): return m * Y_std + Y_mean
def unscale_var(v, Y_std):          return v * (Y_std ** 2)


# ================================================================
# ===== median heuristic =========================================
# ================================================================

def median_heuristic(X):
    dists = pairwise_distances(X).flatten()
    dists = dists[dists > 0]
    return float(np.median(dists))


# ================================================================
# ===== 单次训练 + 评估 ==========================================
# ================================================================

def run_single(M, dataset, mode):
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 65)
    print(f"  VFE GP  |  数据集: {dataset.upper()}  |  M={M}  |  mode={mode}")
    print("=" * 65)

    # ── 1. 加载数据 ──────────────────────────────────────────────
    X_tr, y_tr, X_on, y_on, X_val, y_val, info = load_dataset(dataset)

    # ── 2. 标准化（只用 train 统计量）────────────────────────────
    X_mean, X_std, Y_mean, Y_std = build_scalers(X_tr, y_tr, dataset)
    X_tr_s  = scale_X(X_tr,  X_mean, X_std)
    X_on_s  = scale_X(X_on,  X_mean, X_std)
    X_val_s = scale_X(X_val, X_mean, X_std)
    y_tr_s  = scale_y(y_tr,  Y_mean, Y_std)
    y_on_s  = scale_y(y_on,  Y_mean, Y_std)

    # ── 3. 根据 mode 决定训练数据 ─────────────────────────────────
    if mode == 'train_only':
        X_fit, y_fit = X_tr_s, y_tr_s
    elif mode == 'train_and_online':
        X_fit = np.vstack([X_tr_s, X_on_s])
        y_fit = np.concatenate([y_tr_s, y_on_s])
    else:
        raise ValueError(f"未知 mode: {mode}")

    # ── 4. 打印数据信息 ──────────────────────────────────────────
    print(f"\n训练模式: {mode}")
    print(f"  train: {info['N_train']} 条  |  online: {info['N_online']} 条  "
          f"|  validate: {info['N_test']} 条  |  D={info['D']}")
    print(f"  fit 数据量: {len(X_fit)} 条")
    print(f"  y 分布 (原始尺度):")
    print(f"    train:    mean={y_tr.mean():.4e}, std={y_tr.std():.4e}, "
          f"min={y_tr.min():.4e}, max={y_tr.max():.4e}")
    print(f"    online:   mean={y_on.mean():.4e}, std={y_on.std():.4e}, "
          f"min={y_on.min():.4e}, max={y_on.max():.4e}")
    print(f"    validate: mean={y_val.mean():.4e}, std={y_val.std():.4e}, "
          f"min={y_val.min():.4e}, max={y_val.max():.4e}")

    # ── 5. optimize_Xm 条件判断已移除，始终联合优化 ─────────────

    # ── 6. 初始化模型 ─────────────────────────────────────────────
    ls_init    = median_heuristic(X_fit)
    y_std_s    = float(y_fit.std())
    noise_init = max(y_std_s * 0.1, 1e-4)

    kmeans  = KMeans(n_clusters=M, random_state=42, n_init=10).fit(X_fit)
    Xm_init = kmeans.cluster_centers_

    model = VFEGP(
        Xm_init         = Xm_init,
        noise_var        = noise_init,
        lengthscale_init = ls_init,
        variance_init    = max(y_std_s ** 2, 1e-4),
        jitter           = MODEL_CONFIG['jitter'],
    )

    print(f"\n初始超参: ℓ={ls_init:.4f}, σ²={y_std_s**2:.4f}, noise={noise_init:.5f}")
    print(f"模型配置: M={M}  (联合优化 Xm + 超参数)")

    # ── 7. 训练 ──────────────────────────────────────────────────
    cfg        = TRAIN_CONFIG_MAP[dataset.lower()]
    fv_history = fit(
        model, X_fit, y_fit, cfg,
        verbose        = OUTPUT_CONFIG['verbose'],
        print_interval = OUTPUT_CONFIG['print_interval'],
    )

    # ── 8. 评估 ──────────────────────────────────────────────────
    mean_s, var_s = model.predict(X_val_s)
    mean_orig     = unscale_mean(mean_s, Y_mean, Y_std)
    var_orig      = unscale_var(var_s,   Y_std)
    metrics       = compute_metrics(mean_orig, var_orig, y_val)

    # ── 9. 打印结果 ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  最终结果 — {dataset.upper()}  M={M}  mode={mode}")
    print("=" * 65)
    print(f"  Validate SMSE:     {metrics['smse']:.6f}  （< 1 优于均值基线）")
    print(f"  Validate NLP:      {metrics['nlp']:.6f}")
    print(f"  Validate Coverage: {metrics['coverage']:.4f}  （目标 ≈ 0.95）")
    print(f"\n  收敛超参:")
    for k, v in model.get_hyperparameters().items():
        print(f"    {k:15s}: {v}")

    # ── 10. 保存结果 ──────────────────────────────────────────────
    if OUTPUT_CONFIG['save_stats']:
        save_dir = OUTPUT_CONFIG['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f'vfe_{dataset}_M{M}_{mode}.npz')
        np.savez(
            fname,
            fv_history   = np.array(fv_history),
            smse_val     = metrics['smse'],
            nlp_val      = metrics['nlp'],
            coverage_val = metrics['coverage'],
            R_final      = model.Xm.detach().numpy(),
            lengthscale  = model.get_hyperparameters()['lengthscale'],
            signal_var   = model.get_hyperparameters()['signal_var'],
            noise_var    = model.get_hyperparameters()['noise_var'],
            scaler       = np.array([Y_mean, Y_std, *X_mean, *X_std]),
            num_Xm       = M,
            mode         = mode,
            epochs       = cfg['epochs'],
        )
        print(f"\n结果已保存: {fname}")

    return model, metrics, fv_history


# ================================================================
# ===== 主入口 ====================================================
# ================================================================

def main():
    # key: (dataset, mode, M)
    all_results = {}
    total = len(DATASET_LIST) * len(MODE_LIST) * len(M_LIST)
    idx   = 0

    for dataset in DATASET_LIST:
        for mode in MODE_LIST:
            for M in M_LIST:
                idx += 1
                print(f"\n{'='*65}")
                print(f"  [{idx}/{total}]  dataset={dataset.upper()}  mode={mode}  M={M}")
                print(f"{'='*65}")
                model, metrics, fv_hist = run_single(M, dataset, mode)
                all_results[(dataset, mode, M)] = {'metrics': metrics, 'fv_history': fv_hist}

    # ── 汇总表（按数据集分组）────────────────────────────────────
    for dataset in DATASET_LIST:
        print(f"\n{'='*65}")
        print(f"  汇总 — {dataset.upper()}  (VFE GP)")
        print(f"{'='*65}")
        print(f"  {'mode':<22}  {'M':>4}  {'SMSE':>10}  {'NLP':>10}  {'Coverage':>10}")
        print(f"  {'-'*60}")
        for mode in MODE_LIST:
            for M in M_LIST:
                m = all_results[(dataset, mode, M)]['metrics']
                print(f"  {mode:<22}  {M:>4}  "
                      f"{m['smse']:>10.6f}  {m['nlp']:>10.6f}  {m['coverage']:>10.4f}")
            print(f"  {'-'*60}")

    csv_path = os.path.join(OUTPUT_CONFIG['save_dir'], 'vfe_summary.csv')
    os.makedirs(OUTPUT_CONFIG['save_dir'], exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'M', 'train_only_smse', 'train_and_online_smse'])
        for dataset in DATASET_LIST:
            for M in M_LIST:
                smse_train  = all_results.get((dataset, 'train_only',       M), {}).get('metrics', {}).get('smse', '\\')
                smse_online = all_results.get((dataset, 'train_and_online', M), {}).get('metrics', {}).get('smse', '\\')
                writer.writerow([dataset, M,
                                 f"{smse_train:.6f}"  if isinstance(smse_train,  float) else smse_train,
                                 f"{smse_online:.6f}" if isinstance(smse_online, float) else smse_online])
    print(f"\n  汇总 CSV 已保存: {csv_path}")
    return all_results


if __name__ == '__main__':
    results = main()