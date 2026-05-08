"""
Full GP Runner
==============
基于标准高斯过程（完整 GP），PyTorch 实现。
复杂度 O(N³)，适用于小规模数据集。

训练模式:
  'train_only'      — 只用 train 数据 fit()，validate 评估  → initial_smse
  'train_and_online'— train + online 合并为 full-batch，fit()，validate 评估 → full_smse

单阶段训练：一个 optimizer 同时优化 lengthscale、signal_var、noise_var。
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import pairwise_distances
import pandas as pd


# ================================================================
# ===== 数据集类 ==================================================
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
        index_col = 0 if self.name == 'building' else None
        return pd.read_csv(path, index_col=index_col)
        #return pd.read_csv(path, index_col=None)
    
    def get_splits(self):
        data = self._read_csv(self.file_map[self.name])
        result = {'train': data.iloc[:100]}
        if self.name == 'vanderpol':
            result['online']   = data.iloc[100:1000]
            result['validate'] = data.iloc[1000:]
        elif self.name == 'building':
            result['train']    = data.iloc[:100]
            result['online']   = data.iloc[100:17520]
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
DATASET_LIST = ['building' ]


MODE_LIST  = ['train_only']

# 循环的 epoch 数；设为 [] 则用 TRAIN_CONFIG_MAP 里各数据集的默认值
EPOCH_LIST = [200]

TRAIN_CONFIG_MAP = {
    # 四函数
    'himmelblau':   {'epochs': 500, 'lr': 0.01, 'patience': 50, 'record_time': False},
    'rastrigin':    {'epochs': 500, 'lr': 0.01, 'patience': 50, 'record_time': False},
    'rosenbrock':   {'epochs': 500, 'lr': 0.01, 'patience': 50, 'record_time': False},
    'sixhumpcamel': {'epochs': 500, 'lr': 0.01, 'patience': 50, 'record_time': False},
    # 静态数据集
    'boston':       {'epochs': 500, 'lr': 0.01, 'patience': 50, 'record_time': False},
    'concrete':     {'epochs': 500, 'lr': 0.01, 'patience': 50, 'record_time': False},
    # 系统辨识
    'vanderpol':    {'epochs': 800, 'lr': 0.005, 'patience': 80, 'record_time': False},
    'building':     {'epochs': 800, 'lr': 0.005, 'patience': 80, 'record_time': False},
    'boucwen':      {'epochs': 800, 'lr': 0.005, 'patience': 80, 'record_time': False},
    'tanks':        {'epochs': 800, 'lr': 0.005, 'patience': 80, 'record_time': False},
}

MODEL_CONFIG = {
    'jitter': 1e-6,
}

OUTPUT_CONFIG = {
    'save_dir':       r'D:\project\FullGP_result',
    'save_stats':     True,
    'verbose':        True,
    'print_interval': 50,
}

FUNC_DATASETS   = {'himmelblau', 'rastrigin', 'rosenbrock', 'sixhumpcamel'}
STATIC_DATASETS = {'boston', 'concrete'}
SYSID_DATASETS  = {'vanderpol', 'building', 'boucwen', 'tanks'}


# ================================================================
# ===== Full GP 模型 ==============================================
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
        ls  = torch.clamp(torch.exp(self.log_lengthscale), min=1e-3, max=20.0)
        var = torch.exp(self.log_variance)
        dist_sq = torch.cdist(x1 / ls, x2 / ls, p=2) ** 2
        return var * torch.exp(-0.5 * dist_sq)

    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale).detach().numpy()

    @property
    def signal_variance(self):
        return torch.exp(self.log_variance).item()


class FullGP:
    def __init__(self, input_dim, lengthscale_init=1.0, variance_init=1.0,
                 noise_var=0.1, jitter=1e-6):
        self.jitter  = jitter
        self.kernel  = RBFKernel(input_dim, lengthscale_init, variance_init)
        self.log_beta = nn.Parameter(
            torch.tensor(np.log(1.0 / noise_var), dtype=torch.float64)
        )
        self._X_train = None
        self._alpha   = None
        self._L       = None

    def parameters(self):
        return list(self.kernel.parameters()) + [self.log_beta]

    @staticmethod
    def _chol(A):
        try:
            return torch.linalg.cholesky(A)
        except RuntimeError:
            n = A.shape[0]
            return torch.linalg.cholesky(
                A + torch.eye(n, dtype=torch.float64) * 1e-4
            )

    def _Knn(self, X):
        beta  = torch.exp(self.log_beta)
        noise = 1.0 / beta
        K     = self.kernel(X, X)
        return K + (noise + self.jitter) * torch.eye(len(X), dtype=torch.float64)

    def compute_loss(self, X, y):
        N   = X.shape[0]
        Knn = self._Knn(X)
        L   = self._chol(Knn)
        alpha  = torch.cholesky_solve(y, L)
        quad   = (y.T @ alpha).squeeze()
        log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
        log_lik = (
            -0.5 * quad
            - 0.5 * log_det
            - 0.5 * N * torch.log(torch.tensor(2 * np.pi, dtype=torch.float64))
        )
        return -log_lik, log_lik.item()

    @torch.no_grad()
    def _compute_posterior(self, X, y):
        Knn         = self._Knn(X)
        self._L     = self._chol(Knn)
        self._alpha = torch.cholesky_solve(y, self._L)
        self._X_train = X.clone()

    @torch.no_grad()
    def predict(self, X_star):
        if self._alpha is None:
            raise RuntimeError("请先调用 fit()。")
        X_s   = torch.as_tensor(X_star, dtype=torch.float64)
        beta  = torch.exp(self.log_beta)
        noise = 1.0 / beta
        Ks    = self.kernel(X_s, self._X_train)
        Kss_d = torch.diagonal(self.kernel(X_s, X_s))
        mean  = (Ks @ self._alpha).squeeze(-1)
        V     = torch.linalg.solve_triangular(self._L, Ks.T, upper=False)
        var   = Kss_d - torch.sum(V ** 2, dim=0) + noise
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
    单阶段全批次训练，带早停。
    返回 (lml_history, train_time_sec)。
    """
    X = torch.as_tensor(X_train, dtype=torch.float64)
    y = torch.as_tensor(y_train, dtype=torch.float64).reshape(-1, 1)

    epochs      = cfg['epochs']
    lr          = cfg['lr']
    patience    = cfg.get('patience', 50)
    record_time = cfg.get('record_time', False)

    lml_history = []
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.StepLR(
        opt, step_size=max(1, epochs // 3), gamma=0.5
    )

    best = -np.inf
    wait = 0

    t_start = time.perf_counter() if record_time else None

    for ep in range(epochs):
        opt.zero_grad()
        loss, lml = model.compute_loss(X, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        sch.step()
        lml_history.append(lml)

        if verbose and ep % print_interval == 0:
            hp   = model.get_hyperparameters()
            ls   = hp['lengthscale']
            ls_s = f"{ls.round(4)}" if ls.size > 1 else f"{ls[0]:.4f}"
            print(f"  [{ep:4d}/{epochs}] LML={lml:10.3f} | "
                  f"ℓ={ls_s}, σ²={hp['signal_var']:.4f}, "
                  f"noise={hp['noise_var']:.5f}")

        if patience > 0:
            if lml > best + 1e-4:
                best = lml
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"  Early stop at epoch {ep}")
                    break

    train_time = (time.perf_counter() - t_start) if record_time else None

    if verbose:
        print("\nComputing posterior (caching L and alpha)...")
    model._compute_posterior(X, y)

    if verbose:
        print(f"Done.  Final LML = {lml_history[-1]:.3f}")
        for k, v in model.get_hyperparameters().items():
            print(f"  {k:15s}: {v}")
        if record_time:
            print(f"  训练时间:       {train_time:.2f} 秒")
        print("=" * 65)

    return lml_history, train_time


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

def build_scalers(X_train, y_train):
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

def scale_X(X, X_mean, X_std):        return (X - X_mean) / X_std
def scale_y(y, Y_mean, Y_std):        return (y - Y_mean) / Y_std
def unscale_mean(m, Y_mean, Y_std):   return m * Y_std + Y_mean
def unscale_var(v, Y_std):            return v * (Y_std ** 2)


# ================================================================
# ===== Median heuristic =========================================
# ================================================================

def median_heuristic(X):
    dists = pairwise_distances(X).flatten()
    dists = dists[dists > 0]
    return float(np.median(dists))


# ================================================================
# ===== 单次训练 + 评估 ==========================================
# ================================================================

def run_single(dataset, mode):
    """
    mode='train_only'       → initial_smse（仅 train 数据训练）
    mode='train_and_online' → full_smse   （train+online 合并训练）
    两种 mode 都对同一 validate 集评估。
    返回: (model, metrics, lml_history)
    """
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 65)
    print(f"  Full GP  |  数据集: {dataset.upper()}  |  mode={mode}")
    print("=" * 65)

    # ── 1. 加载数据 ──────────────────────────────────────────────
    X_tr, y_tr, X_on, y_on, X_val, y_val, info = load_dataset(dataset)

    # ── 2. 标准化（scaler 始终由 train 数据建立）────────────────
    X_mean, X_std, Y_mean, Y_std = build_scalers(X_tr, y_tr)
    X_tr_s  = scale_X(X_tr,  X_mean, X_std)
    X_on_s  = scale_X(X_on,  X_mean, X_std)
    X_val_s = scale_X(X_val, X_mean, X_std)
    y_tr_s  = scale_y(y_tr,  Y_mean, Y_std)
    y_on_s  = scale_y(y_on,  Y_mean, Y_std)

    # ── 3. 训练数据选择 ───────────────────────────────────────────
    if mode == 'train_only':
        X_fit, y_fit = X_tr_s, y_tr_s
    elif mode == 'train_and_online':
        X_fit = np.vstack([X_tr_s, X_on_s])
        y_fit = np.concatenate([y_tr_s, y_on_s])
    else:
        raise ValueError(f"未知 mode: {mode}")

    # ── 4. 打印数据信息 ──────────────────────────────────────────
    print(f"\n训练模式: {mode}")
    print(f"  train:    {info['N_train']} 条")
    print(f"  online:   {info['N_online']} 条")
    print(f"  validate: {info['N_test']} 条  |  D={info['D']}")
    print(f"  fit 数据量: {len(X_fit)} 条")
    print(f"  y 分布 (原始尺度):")
    print(f"    train:    mean={y_tr.mean():.2f}, std={y_tr.std():.2f}, "
          f"min={y_tr.min():.2f}, max={y_tr.max():.2f}")
    print(f"    online:   mean={y_on.mean():.2f}, std={y_on.std():.2f}, "
          f"min={y_on.min():.2f}, max={y_on.max():.2f}")
    print(f"    validate: mean={y_val.mean():.2f}, std={y_val.std():.2f}, "
          f"min={y_val.min():.2f}, max={y_val.max():.2f}")

    # ── 5. 初始化模型 ─────────────────────────────────────────────
    ls_init    = median_heuristic(X_fit)
    y_std_s    = float(y_fit.std())
    noise_init = max(y_std_s * 0.1, 1e-4)

    model = FullGP(
        input_dim        = info['D'],
        lengthscale_init = ls_init,
        variance_init    = max(y_std_s ** 2, 1e-4),
        noise_var        = noise_init,
        jitter           = MODEL_CONFIG['jitter'],
    )
    print(f"\n初始超参: ℓ={ls_init:.4f}, σ²={y_std_s**2:.4f}, noise={noise_init:.5f}")

    # ── 6. 训练 ──────────────────────────────────────────────────
    cfg = TRAIN_CONFIG_MAP[dataset.lower()]
    print("=" * 65)
    print(f"训练: epochs={cfg['epochs']}, lr={cfg['lr']}, patience={cfg['patience']}")
    print("=" * 65)

    lml_history, train_time = fit(
        model, X_fit, y_fit, cfg,
        verbose        = OUTPUT_CONFIG['verbose'],
        print_interval = OUTPUT_CONFIG['print_interval'],
    )

    # ── 7. 评估 ──────────────────────────────────────────────────
    mean_s, var_s = model.predict(X_val_s)
    mean_orig     = unscale_mean(mean_s, Y_mean, Y_std)
    var_orig      = unscale_var(var_s,   Y_std)
    metrics       = compute_metrics(mean_orig, var_orig, y_val)

    # ── 8. 打印结果 ───────────────────────────────────────────────
    mode_label = 'initial' if mode == 'train_only' else 'full'
    print("\n" + "=" * 65)
    print(f"  最终结果 — {dataset.upper()}  mode={mode}  [{mode_label}]")
    print("=" * 65)
    print(f"  Validate SMSE     ({mode_label}): {metrics['smse']:.6f}")
    print(f"  Validate NLP      ({mode_label}): {metrics['nlp']:.6f}")
    print(f"  Validate Coverage ({mode_label}): {metrics['coverage']:.4f}")
    if train_time is not None:
        print(f"  训练时间:          {train_time:.2f} 秒")
    print(f"\n  收敛超参:")
    for k, v in model.get_hyperparameters().items():
        print(f"    {k:15s}: {v}")

    # ── 9. 保存 .npz ──────────────────────────────────────────────
    if OUTPUT_CONFIG['save_stats']:
        save_dir = OUTPUT_CONFIG['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f'fullgp_{dataset}_{mode}.npz')
        save_dict = dict(
            lml_history  = np.array(lml_history),
            smse_val     = metrics['smse'],
            nlp_val      = metrics['nlp'],
            coverage_val = metrics['coverage'],
            lengthscale  = model.get_hyperparameters()['lengthscale'],
            signal_var   = model.get_hyperparameters()['signal_var'],
            noise_var    = model.get_hyperparameters()['noise_var'],
            scaler       = np.array([Y_mean, Y_std, *X_mean, *X_std]),
            mode         = mode,
            epochs       = cfg['epochs'],
            N_fit        = len(X_fit),
        )
        if train_time is not None:
            save_dict['train_time_sec'] = train_time
        np.savez(fname, **save_dict)
        print(f"\n结果已保存: {fname}")

    return model, metrics, lml_history


# ================================================================
# ===== 主入口 ====================================================
# ================================================================

def main():
    all_results = {}
    # key: (dataset, mode, ep_override) → {'metrics': {...}, 'lml_history': [...]}

    for dataset in DATASET_LIST:
        epoch_list = EPOCH_LIST if EPOCH_LIST else [TRAIN_CONFIG_MAP[dataset.lower()]['epochs']]
        total = len(MODE_LIST) * len(epoch_list)
        idx   = 0

        for mode in MODE_LIST:
            for ep_override in epoch_list:
                idx += 1
                print(f"\n{'='*65}")
                print(f"  [{idx}/{total}]  dataset={dataset.upper()}  "
                      f"mode={mode}  epochs={ep_override}")
                print(f"{'='*65}")

                # 临时覆盖 epoch，不影响其他数据集
                cfg_orig = TRAIN_CONFIG_MAP[dataset.lower()].copy()
                TRAIN_CONFIG_MAP[dataset.lower()]['epochs'] = ep_override

                model, metrics, lml_hist = run_single(dataset, mode)
                all_results[(dataset, mode, ep_override)] = {
                    'metrics':     metrics,
                    'lml_history': lml_hist,
                }

                TRAIN_CONFIG_MAP[dataset.lower()] = cfg_orig

    # ── 控制台汇总表 ─────────────────────────────────────────────
    COL = 10
    print(f"\n{'='*95}")
    print(f"  汇总 — Full GP  "
          f"({'  '.join(DATASET_LIST)})")
    print(f"{'='*95}")
    print(f"  {'dataset':<12}  {'epochs':>6}  "
          f"{'init_SMSE':>{COL}}  {'init_NLP':>{COL}}  {'init_COV':>{COL}}  "
          f"{'full_SMSE':>{COL}}  {'full_NLP':>{COL}}  {'full_COV':>{COL}}")
    print(f"  {'-'*93}")

    for dataset in DATASET_LIST:
        epoch_list = EPOCH_LIST if EPOCH_LIST else [TRAIN_CONFIG_MAP[dataset.lower()]['epochs']]
        for ep_override in epoch_list:
            # 取 train_only 结果（initial）
            r_init = all_results.get((dataset, 'train_only', ep_override), {}).get('metrics')
            # 取 train_and_online 结果（full）
            r_full = all_results.get((dataset, 'train_and_online', ep_override), {}).get('metrics')

            def fmt(r, key, fmt_str):
                return f"{r[key]:{fmt_str}}" if r else '\\'

            print(f"  {dataset:<12}  {ep_override:>6}  "
                  f"{fmt(r_init, 'smse',     '>10.6f')}  "
                  f"{fmt(r_init, 'nlp',      '>10.6f')}  "
                  f"{fmt(r_init, 'coverage', '>10.4f')}  "
                  f"{fmt(r_full, 'smse',     '>10.6f')}  "
                  f"{fmt(r_full, 'nlp',      '>10.6f')}  "
                  f"{fmt(r_full, 'coverage', '>10.4f')}")
        print(f"  {'-'*93}")

    # ── 保存汇总 CSV ─────────────────────────────────────────────
    if OUTPUT_CONFIG['save_stats']:
        save_dir = OUTPUT_CONFIG['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, 'fullgp_summary.csv')

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'dataset', 'epochs',
                'initial_smse', 'initial_nlp', 'initial_coverage',
                'full_smse',    'full_nlp',    'full_coverage',
            ])
            for dataset in DATASET_LIST:
                epoch_list = (EPOCH_LIST if EPOCH_LIST
                              else [TRAIN_CONFIG_MAP[dataset.lower()]['epochs']])
                for ep_override in epoch_list:
                    r_init = all_results.get(
                        (dataset, 'train_only',       ep_override), {}).get('metrics')
                    r_full = all_results.get(
                        (dataset, 'train_and_online', ep_override), {}).get('metrics')

                    def csv_val(r, key, fmt_str):
                        return f"{r[key]:{fmt_str}}" if r else '\\'

                    writer.writerow([
                        dataset,
                        ep_override,
                        csv_val(r_init, 'smse',     '.6f'),
                        csv_val(r_init, 'nlp',      '.6f'),
                        csv_val(r_init, 'coverage', '.4f'),
                        csv_val(r_full, 'smse',     '.6f'),
                        csv_val(r_full, 'nlp',      '.6f'),
                        csv_val(r_full, 'coverage', '.4f'),
                    ])

        print(f"\n  汇总 CSV 已保存: {csv_path}")

    return all_results


if __name__ == '__main__':
    results = main()