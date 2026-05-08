"""
RGP Runner (Recursive Gaussian Process)
========================================
基于 Huber (2013/2014) Recursive GP 算法。对齐 SVI-GP Runner 框架。

Xm初始化后不更新,
只更新超参数,基于损失函数MLE

训练阶段:
  Phase 1 (offline) — 只用 train data:
    - 可选: MLE 超参数优化（hp_learn_phase1 开关, 默认 False）
    - 用 train data 逐点 update 热身 basis vectors (μ, Σ)
    - Xb 用 KMeans 聚类中心初始化（只用 train data）

  Phase 2 (online) — 顺序 mini-batch:
    - predict-before-update → 记录每步 MSE, 用于 online SMSE 汇总
    - 逐点 update model (μ, Σ)
    - 可选: 每步重新 MLE 优化超参数（update_hp_online 开关, 默认 True）

Online SMSE 计算:
  与 SVI runner 一致: predict before update → 累积误差
  → online_smse = mean(all_batch_MSE) / var(y_online_orig)

注意（HP 更新的数值一致性）:
  每步在线 MLE 后重新设置 kernel 和 noise_var、重算 Lbb，
  但 μ/Σ 保持不变作为近似（online GP 的标准做法）。
  如果需要精确一致性，可将 hp_learn_phase1=True 并关闭 update_hp_online。
"""

import os
import csv
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve, solve_triangular
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import pandas as pd

# ================================================================
# ===== SEKernel, HpLearning, RecursiveGP (standalone) ===========
# ================================================================

class SEKernel:
    def __init__(self, lengthscale, variance):
        self.lengthscale = float(lengthscale)
        self.variance    = float(variance)

    def __call__(self, input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
        inv_l = 1.0 / self.lengthscale
        a     = input1 * inv_l
        b     = input2 * inv_l
        a2    = np.sum(a ** 2, axis=1, keepdims=True)
        b2    = np.sum(b ** 2, axis=1, keepdims=True)
        ab    = a @ b.T
        return self.variance * np.exp(-0.5 * (a2 + b2.T - 2 * ab))


class HpLearning:
    """
    最大化 log 边际似然，估计 (lengthscale, signal_var, noise_var)。

    mle_mode 控制用哪些数据做 MLE：
      'batch_only' — 只用最新 batch 真实观测，O(B³)，数学最严格（默认）
                     y 与 θ 无循环依赖；代价是不含历史超参约束
      'xb_batch'   — Xb + μ 拼接最新 batch 真实观测，O((M+B)³)
                     y_batch 部分数学严格，μ 部分近似；历史与新数据兼顾
      'xb_only'    — 只用 Xb + μ（后验均值作为伪观测），O(M³)，最快
                     纯启发式，μ 与 θ 存在循环依赖
      'full'       — 全量累积数据，数学严格但 O(N³)，仅供对比
    """

    def __init__(
        self,
        X        : np.ndarray,
        y        : np.ndarray,
        jitter   : float      = 1e-6,
        mle_mode : str        = 'batch_only',
        Xb       : np.ndarray = None,
        mu       : np.ndarray = None,   # shape (M,1) 或 (M,)
        X_batch  : np.ndarray = None,   # 最新 batch X
        y_batch  : np.ndarray = None,   # 最新 batch y
    ):
        self.mle_mode = mle_mode
        self.jitter   = jitter

        if mle_mode == 'batch_only':
            if X_batch is None or y_batch is None:
                raise ValueError("mle_mode='batch_only' 需要传入 X_batch 和 y_batch")
            self.X = np.atleast_2d(X_batch).astype(float)
            self.y = np.atleast_1d(y_batch).ravel().astype(float)

        elif mle_mode == 'xb_batch':
            if Xb is None or mu is None or X_batch is None or y_batch is None:
                raise ValueError("mle_mode='xb_batch' 需要传入 Xb, mu, X_batch, y_batch")
            self.X = np.vstack([np.atleast_2d(Xb), np.atleast_2d(X_batch)]).astype(float)
            self.y = np.concatenate([
                np.atleast_1d(mu).ravel(),
                np.atleast_1d(y_batch).ravel()
            ]).astype(float)

        elif mle_mode == 'xb_only':
            if Xb is None or mu is None:
                raise ValueError("mle_mode='xb_only' 需要传入 Xb 和 mu")
            self.X = np.atleast_2d(Xb).astype(float)
            self.y = np.atleast_1d(mu).ravel().astype(float)

        elif mle_mode == 'full':
            self.X = np.atleast_2d(X).astype(float)
            self.y = np.atleast_1d(y).ravel().astype(float)

        else:
            raise ValueError(
                f"未知 mle_mode: {mle_mode}，"
                f"可选 'batch_only' / 'xb_batch' / 'xb_only' / 'full'"
            )

        self.n = len(self.y)

    def _nll(self, params: np.ndarray) -> float:
        try:
            ls, sv, nv = np.exp(params)
            if not (np.isfinite(ls) and np.isfinite(sv) and np.isfinite(nv)):
                return 1e10
            kernel = SEKernel(lengthscale=ls, variance=sv)
            K = kernel(self.X, self.X) + np.eye(self.n) * (nv + self.jitter)
            try:
                L = cholesky(K, lower=True)
            except np.linalg.LinAlgError:
                return 1e10
            alpha = cho_solve((L, True), self.y)
            lml   = (-0.5 * self.y @ alpha
                     - np.sum(np.log(np.diag(L)))
                     - 0.5 * self.n * np.log(2 * np.pi))
            return -lml if np.isfinite(lml) else 1e10
        except Exception:
            return 1e10

    def optimize(
        self,
        init_guess: list,
        bounds    : list,
        max_iter  : int = 200,
    ) -> np.ndarray:
        """返回 [ls, sv, nv]（原始空间）。"""
        # 守卫：确保 init_guess 全部正数有限
        safe_init = [
            v if (np.isfinite(v) and v > 1e-10) else fb
            for v, fb in zip(init_guess, [1.0, 1.0, 1e-3])
        ]
        res = minimize(
            self._nll,
            np.log(safe_init),
            method ='L-BFGS-B',
            bounds  = bounds,
            options = {'ftol': 1e-7, 'maxiter': max_iter},
        )
        opt    = np.exp(res.x)
        status = "converged" if res.success else "NOT converged (using current)"
        print(f"    [MLE/{self.mle_mode}] {status}: "
              f"ls={opt[0]:.4f}, sv={opt[1]:.4f}, nv={opt[2]:.6f}  "
              f"(n={self.n})")
        return opt  # [ls, sv, nv]


class RecursiveGP:
    """
    Huber (2014) Algorithm 1: Recursive Gaussian Process (RGP).

    维护 basis-vector 后验 p(g | y_{1:t}) = N(μ, Σ) 并递归更新。
    预测时通过 GP 推断 (inference step) 计算任意测试点的后验均值/方差。
    """

    def __init__(
        self,
        kernel_input: SEKernel,
        Xb          : np.ndarray,
        noise_var   : float,
        jitter      : float = 1e-9,
    ):
        self.Xb        = np.atleast_2d(Xb)
        self.kernel    = kernel_input
        self.noise_var = float(noise_var)
        self.jitter    = float(jitter)

        Kbb        = self.kernel(self.Xb, self.Xb) + np.eye(self.Xb.shape[0]) * self.jitter
        self.Lbb   = cholesky(Kbb, lower=True)
        self.mu    = np.zeros((self.Xb.shape[0], 1))
        self.Sigma = Kbb.copy()

    # ── 内部工具 ─────────────────────────────────────────────────

    def _Kbb_solve(self, B: np.ndarray) -> np.ndarray:
        return cho_solve((self.Lbb, True), B)

    def _Jt(self, X_t: np.ndarray) -> np.ndarray:
        """Eq.(8): J_t = k(X_t, X) · k(X,X)^{-1}"""
        Ktb = self.kernel(X_t, self.Xb)
        return Ktb @ self._Kbb_solve(np.eye(self.Xb.shape[0]))

    def _prior_at(self, X_t: np.ndarray):
        """
        Inference step (Eq.6-9):
        returns (μ_p, C_p, J) — prior mean, prior cov, gain matrix
        """
        X_t  = np.atleast_2d(X_t)
        J    = self._Jt(X_t)
        ktt  = self.kernel(X_t, X_t)
        kbt  = self.kernel(self.Xb, X_t)
        B    = ktt - J @ kbt
        mu_p = J @ self.mu
        C_p  = B + J @ self.Sigma @ J.T
        return mu_p, C_p, J

    # ── 递归更新 ─────────────────────────────────────────────────

    def update(self, X_t: np.ndarray, y_t: np.ndarray) -> None:
        """
        Algorithm 1, Steps 3-4: Kalman-style update of (μ, Σ).
        Eq.(10-12).
        """
        X_t  = np.atleast_2d(X_t)
        y_t  = y_t.reshape(-1, 1) if y_t.ndim == 1 else y_t
        mu_p, C_p, J = self._prior_at(X_t)

        S     = C_p + np.eye(C_p.shape[0]) * self.noise_var
        S_inv = np.linalg.solve(S, np.eye(S.shape[0]))
        G     = self.Sigma @ J.T @ S_inv          # G̃_t, Eq.(12)

        self.mu    = self.mu + G @ (y_t - mu_p)   # Eq.(10)
        Sigma_new  = self.Sigma - G @ J @ self.Sigma  # Eq.(11)
        # 强制对称，防止数值误差积累
        self.Sigma = 0.5 * (Sigma_new + Sigma_new.T)

    # ── 预测 ─────────────────────────────────────────────────────

    def predict(self, Xstar: np.ndarray, noisy: bool = False):
        """
        后验预测均值和方差。
        noisy=True 时加观测噪声 σ²（对应 y* 的预测方差）。
        """
        Xstar        = np.atleast_2d(Xstar)
        mu_p, C_p, _ = self._prior_at(Xstar)
        mean         = mu_p[:, 0]
        var          = np.clip(np.diag(C_p), 0.0, None)
        if noisy:
            var = var + self.noise_var
        return mean, var

    # ── HP 更新后重建 Lbb ─────────────────────────────────────────

    def rebuild_lbb(self) -> None:
        """
        在 kernel / noise_var 被外部修改后，
        重新计算 Lbb（μ/Σ 保持不变，作为近似）。
        """
        Kbb      = self.kernel(self.Xb, self.Xb) + np.eye(self.Xb.shape[0]) * self.jitter
        self.Lbb = cholesky(Kbb, lower=True)


# ================================================================
# ===== Dataset classes ==================
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

#DATASET_LIST = ['rastrigin','rosenbrock','himmelblau','sixhumpcamel','boston','concrete','vanderpol','boucwen','tanks']
# 四函数:   'rastrigin' , 'rosenbrock' , 'himmelblau' , 'sixhumpcamel'
# 静态:     'boston'    , 'concrete'
# 系统辨识: 'vanderpol' , 'boucwen', 'tanks', 'building'

DATASET_LIST = ['building']
M_LIST       = [50]

# ── 训练配置 ────────────────────────────────────────────────────
# online_batch_size : Phase 2 每步处理的样本数
# update_hp_online  : Phase 2 每步是否重新 MLE 优化超参数
# mle_mode          : 在线 MLE 使用的数据模式
#   'batch_only' — 只用最新 batch 真实观测，O(B³)，数学最严格
#   'xb_batch'   — Xb+μ 拼最新 batch，O((M+B)³)，历史与新数据兼顾
#   'xb_only'    — 只用 Xb+μ，O(M³)，最快，纯启发式
#   'full'       — 全量累积，O(N³)，最慢，仅供对比
TRAIN_CONFIG_MAP = {
    'rastrigin':    {'online_batch_size': 100, 'update_hp_online': True, 'mle_mode': 'batch_only'},
    'rosenbrock':   {'online_batch_size': 100, 'update_hp_online': True, 'mle_mode': 'batch_only'},
    'himmelblau':   {'online_batch_size': 100, 'update_hp_online': True, 'mle_mode': 'batch_only'},
    'sixhumpcamel': {'online_batch_size': 100, 'update_hp_online': True, 'mle_mode': 'batch_only'},
    'boston':       {'online_batch_size': 100, 'update_hp_online': True, 'mle_mode': 'batch_only'},
    'concrete':     {'online_batch_size': 100, 'update_hp_online': True, 'mle_mode': 'batch_only'},
    'vanderpol':    {'online_batch_size': 200, 'update_hp_online': True, 'mle_mode': 'batch_only'},
    'boucwen':      {'online_batch_size': 200, 'update_hp_online': True, 'mle_mode': 'batch_only'},
    'tanks':        {'online_batch_size': 200, 'update_hp_online': True, 'mle_mode': 'batch_only'},
    'building':     {'online_batch_size': 200, 'update_hp_online': True, 'mle_mode': 'batch_only'},
}

# ── HpLearning 每数据集配置 ─────────────────────────────────────
# max_iter : L-BFGS-B 最大迭代次数
HP_CONFIG_MAP = {
    'rastrigin':    {'max_iter': 1},
    'rosenbrock':   {'max_iter': 1},
    'himmelblau':   {'max_iter': 1},
    'sixhumpcamel': {'max_iter': 1},
    'boston':       {'max_iter': 1},
    'concrete':     {'max_iter': 30},
    'vanderpol':    {'max_iter': 1},
    'boucwen':      {'max_iter': 1},
    'tanks':        {'max_iter': 20},
    'building':     {'max_iter': 1},
}

OUTPUT_CONFIG = {
    'save_dir':       r'D:\project\RGP_result',
    'save_stats':     False,
    'verbose':        True,
    'print_interval': 10,
}

MODEL_CONFIG = {
    'hp_learn_phase1': False,
    'jitter':          1e-9,
}

FUNC_DATASETS   = {'himmelblau', 'rastrigin', 'rosenbrock', 'sixhumpcamel'}
STATIC_DATASETS = {'boston', 'concrete'}
SYSID_DATASETS  = {'vanderpol', 'boucwen', 'tanks', 'building'}


# ================================================================
# ===== 标准化 ====================================================
# ================================================================

def build_scalers(X_train, y_train, dataset=''):
    col_std  = X_train.std(axis=0)
    col_mean = X_train.mean(axis=0)
    if True:
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
# ===== 数据加载工具 ==============================================
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

    return X_tr, y_tr, X_on, y_on, X_val, y_val, {
        'name': name, 'D': X_tr.shape[1],
        'N_train': len(X_tr), 'N_online': len(X_on), 'N_test': len(X_val),
    }


# ================================================================
# ===== 自动估计 hp bound =========================================
# ================================================================

def auto_bounds(X_scaled, y_scaled, slack=3.0):
    dists  = pairwise_distances(X_scaled).flatten()
    ls_est = float(np.median(dists[dists > 0]))
    ls_lo  = np.log(ls_est) - slack
    ls_hi  = np.log(ls_est) + slack

    sv_est = float(y_scaled.var())
    sv_lo  = np.log(max(sv_est, 1e-4)) - slack
    sv_hi  = np.log(max(sv_est, 1e-4)) + slack + 2

    nv_lo  = -10.0
    nv_hi  = sv_hi - 1.0

    return [(ls_lo, ls_hi), (sv_lo, sv_hi), (nv_lo, nv_hi)]


# ================================================================
# ===== 评估指标 ==================================================
# ================================================================

def median_heuristic(X):
    dists = pairwise_distances(X).flatten()
    return float(np.median(dists[dists > 0]))


def compute_metrics(mean, var, y_true):
    err2     = (mean - y_true) ** 2
    smse     = float(err2.mean() / np.var(y_true))
    nlp      = float(np.mean(0.5 * err2 / var + 0.5 * np.log(2 * np.pi * var)))
    std      = np.sqrt(var)
    coverage = float(((y_true >= mean - 1.96 * std) & (y_true <= mean + 1.96 * std)).mean())
    return {'smse': smse, 'nlp': nlp, 'coverage': coverage}


# ================================================================
# ===== fit (两阶段训练) ==========================================
# ================================================================

def fit(
    model           : RecursiveGP,
    X_train         : np.ndarray,
    y_train         : np.ndarray,
    X_online        : np.ndarray,
    y_online        : np.ndarray,
    hp_cfg          : dict,
    train_cfg       : dict,
    Y_mean          : float,
    Y_std           : float,
    hp_learn_phase1 : bool = False,
    verbose         : bool = True,
    print_interval  : int  = 10,
):
    online_errors = []
    hp_history    = []

    # ── Phase 1 ──────────────────────────────────────────────────
    if verbose:
        print("=" * 65)
        print("Phase 1 (offline): Basis vector warm-up on train data")
        if hp_learn_phase1:
            print("  [MLE 开关] hp_learn_phase1=True → 先做 MLE")
        print("=" * 65)

    if hp_learn_phase1:
        # Phase 1 用 full 模式：此时 μ 尚未 update，无意义，直接用原始 train data
        hpl = HpLearning(X_train, y_train, mle_mode='full')
        ls_new, sv_new, nv_new = hpl.optimize(
            init_guess=[model.kernel.lengthscale,
                        model.kernel.variance,
                        model.noise_var],
            bounds   = hp_cfg['bounds'],
            max_iter = hp_cfg['max_iter'],
        )
        model.kernel    = SEKernel(lengthscale=ls_new, variance=sv_new)
        model.noise_var = nv_new
        model.rebuild_lbb()
        # 重置 μ/Σ（Phase 1 还未 update，直接重置）
        Kbb         = model.kernel(model.Xb, model.Xb) + np.eye(model.Xb.shape[0]) * model.jitter
        model.mu    = np.zeros((model.Xb.shape[0], 1))
        model.Sigma = Kbb.copy()

    # 逐点 update：热身 basis vectors
    for i in range(len(X_train)):
        model.update(X_train[i:i+1], y_train[i:i+1])

    if verbose:
        print(f"  Phase 1 done | N_train={len(X_train)} "
              f"| ls={model.kernel.lengthscale:.4f}, "
              f"sv={model.kernel.variance:.4f}, "
              f"nv={model.noise_var:.6f}")

    # ── Phase 2 ──────────────────────────────────────────────────
    if verbose:
        print()
        print("=" * 65)
        print("Phase 2 (online): predict-before-update, sequential mini-batch")
        print(f"  update_hp_online={train_cfg['update_hp_online']}, "
              f"batch_size={train_cfg['online_batch_size']}, "
              f"mle_mode={train_cfg['mle_mode']}")
        print("=" * 65)

    batch_size       = train_cfg['online_batch_size']
    update_hp_online = train_cfg['update_hp_online']
    mle_mode         = train_cfg['mle_mode']

    # full 模式才需要维护累积数据，其他模式不用
    X_seen = X_train.copy() if mle_mode == 'full' else None
    y_seen = y_train.copy() if mle_mode == 'full' else None

    start = 0
    step  = 0

    while True:
        Xb_np = X_online[start: start + batch_size]
        if len(Xb_np) == 0:
            break
        yb_np = y_online[start: start + batch_size]
        B     = len(Xb_np)

        # Step 1: predict-before-update（原始尺度）
        mean_s, var_s = model.predict(Xb_np, noisy=True)
        mean_orig     = unscale_mean(mean_s, Y_mean, Y_std)
        y_orig        = yb_np * Y_std + Y_mean
        online_errors.append(float(np.mean((mean_orig - y_orig) ** 2)))

        # Step 2: 逐点 update model
        for i in range(B):
            model.update(Xb_np[i:i+1], yb_np[i:i+1])

        # Step 3: [可选] 在线 MLE 超参数重优化
        if update_hp_online:
            # full 模式：累积所有已见数据
            if mle_mode == 'full':
                X_seen = np.vstack([X_seen, Xb_np])
                y_seen = np.concatenate([y_seen, yb_np])

            hpl = HpLearning(
                X        = X_seen,    # full 模式使用；其他模式忽略此参数
                y        = y_seen,
                mle_mode = mle_mode,
                Xb       = model.Xb,
                mu       = model.mu,
                X_batch  = Xb_np,     # batch_only / xb_batch 模式使用
                y_batch  = yb_np,     # batch_only / xb_batch 模式使用
            )
            ls_new, sv_new, nv_new = hpl.optimize(
                init_guess=[model.kernel.lengthscale,
                            model.kernel.variance,
                            model.noise_var],
                bounds   = hp_cfg['bounds'],
                max_iter = hp_cfg['max_iter'],
            )
            # 写回模型（μ/Σ 保持不变，在线 GP 标准近似）
            model.kernel    = SEKernel(lengthscale=ls_new, variance=sv_new)
            model.noise_var = nv_new
            model.rebuild_lbb()
            hp_history.append((ls_new, sv_new, nv_new))
        else:
            hp_history.append((model.kernel.lengthscale,
                               model.kernel.variance,
                               model.noise_var))

        start += batch_size
        step  += 1

        if verbose and step % print_interval == 0:
            y_seen_orig     = y_online[:start] * Y_std + Y_mean
            online_smse_now = (
                float(np.mean(online_errors) / np.var(y_seen_orig))
                if len(y_seen_orig) > 1 else float('nan')
            )
            print(f"  [step {step:4d}]  "
                  f"online SMSE(so far)={online_smse_now:.6f}  |  "
                  f"ls={model.kernel.lengthscale:.4f}  "
                  f"sv={model.kernel.variance:.4f}  "
                  f"nv={model.noise_var:.6f}")

    if verbose:
        print(f"  Phase 2 done | steps={step} | N_online={len(X_online)}")

    return online_errors, hp_history


# ================================================================
# ===== run_single ===============================================
# ================================================================

def run_single(M: int, dataset: str):
    np.random.seed(42)

    print(f"\n{'='*65}")
    print(f"  RGP  |  数据集: {dataset.upper()}  |  M={M}")
    print(f"{'='*65}")

    # ── 1. 加载数据 ──────────────────────────────────────────────
    X_tr, y_tr, X_on, y_on, X_val, y_val, info = load_dataset(dataset)

    # ── 2. 标准化（只用 train 统计量，online data 视为未知）───────
    X_mean, X_std, Y_mean, Y_std = build_scalers(X_tr, y_tr, dataset=dataset)
    X_tr_s  = scale_X(X_tr,  X_mean, X_std)
    X_on_s  = scale_X(X_on,  X_mean, X_std)
    X_val_s = scale_X(X_val, X_mean, X_std)
    y_tr_s  = scale_y(y_tr,  Y_mean, Y_std)
    y_on_s  = scale_y(y_on,  Y_mean, Y_std)

    # ── 3. 打印数据信息 ──────────────────────────────────────────
    print(f"\n  train: {info['N_train']}  |  "
          f"online: {info['N_online']}  |  "
          f"validate: {info['N_test']}  |  D={info['D']}")
    print("  train y:    mean={:.2f}, std={:.2f}, min={:.2f}, max={:.2f}".format(
          y_tr.mean(), y_tr.std(), y_tr.min(), y_tr.max()))
    print("  online y:   mean={:.2f}, std={:.2f}, min={:.2f}, max={:.2f}".format(
          y_on.mean(), y_on.std(), y_on.min(), y_on.max()))
    print("  validate y: mean={:.2f}, std={:.2f}, min={:.2f}, max={:.2f}".format(
          y_val.mean(), y_val.std(), y_val.min(), y_val.max()))
    print(f"  Y_mean={Y_mean:.4f}, Y_std={Y_std:.4f}")
    print(f"  y_tr_s: mean={y_tr_s.mean():.4f}, std={y_tr_s.std():.4f}")
    print(f"  X_mean={X_mean.round(4)},  X_std={X_std.round(4)}")

    # ── 4. 初始化超参数（只用 train data 统计量）─────────────────
    bounds = auto_bounds(X_tr_s, y_tr_s)
    hp_cfg = {**HP_CONFIG_MAP[dataset], 'bounds': bounds}
    ls_init    = median_heuristic(X_tr_s)
    y_std_s    = float(y_tr_s.std())
    noise_init = max(y_std_s * 0.1, 1e-4)
    sv_init    = max(y_std_s ** 2, 1e-4)
    nv_init    = noise_init

    # ── 5. 初始化 Xb（KMeans on train data only）─────────────────
    km        = KMeans(n_clusters=M, random_state=42, n_init=10).fit(X_tr_s)
    Xb_init   = km.cluster_centers_.astype(np.float64)
    train_cfg = TRAIN_CONFIG_MAP[dataset.lower()]

    kernel = SEKernel(lengthscale=ls_init, variance=sv_init)
    model  = RecursiveGP(
        kernel_input=kernel,
        Xb          =Xb_init,
        noise_var   =nv_init,
        jitter      =MODEL_CONFIG['jitter'],
    )

    print(f"\n  初始超参: ls={ls_init:.4f}, sv={sv_init:.4f}, nv={nv_init:.6f}")
    print(f"  模型配置: M={M}, hp_learn_phase1={MODEL_CONFIG['hp_learn_phase1']}, "
          f"update_hp_online={train_cfg['update_hp_online']}, "
          f"online_batch_size={train_cfg['online_batch_size']}, "
          f"mle_mode={train_cfg['mle_mode']}")

    # ── 6. 训练 ──────────────────────────────────────────────────
    online_errors, hp_history = fit(
        model           =model,
        X_train         =X_tr_s,  y_train  =y_tr_s,
        X_online        =X_on_s,  y_online =y_on_s,
        hp_cfg          =hp_cfg,  train_cfg=train_cfg,
        Y_mean          =Y_mean,  Y_std    =Y_std,
        hp_learn_phase1 =MODEL_CONFIG['hp_learn_phase1'],
        verbose         =OUTPUT_CONFIG['verbose'],
        print_interval  =OUTPUT_CONFIG['print_interval'],
    )

    # ── 7. 评估 ──────────────────────────────────────────────────
    online_smse = (
        float(np.mean(online_errors) / np.var(y_on))
        if len(online_errors) > 0 else float('nan')
    )

    mean_s, var_s = model.predict(X_val_s, noisy=True)
    mean_orig     = unscale_mean(mean_s, Y_mean, Y_std)
    var_orig      = unscale_var(var_s, Y_std)
    metrics_val   = compute_metrics(mean_orig, var_orig, y_val)

    # ── 8. 打印结果 ───────────────────────────────────────────────
    print(f"\n  收敛超参:")
    print(f"    lengthscale : {model.kernel.lengthscale:.6f}")
    print(f"    signal_var  : {model.kernel.variance:.6f}")
    print(f"    noise_var   : {model.noise_var:.8f}")
    print(f"\n  Online  SMSE (predict-before-update): {online_smse:.6f}")
    print(f"  Validate SMSE    : {metrics_val['smse']:.6f}  (< 1 优于均值基线)")
    print(f"  Validate NLP     : {metrics_val['nlp']:.6f}")
    print(f"  Validate Coverage: {metrics_val['coverage']:.4f}  (目标 ≈ 0.95)")

    # ── 9. 保存 NPZ ───────────────────────────────────────────────
    if OUTPUT_CONFIG['save_stats']:
        save_dir = OUTPUT_CONFIG['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        fname  = os.path.join(save_dir, f'rgp_{dataset}_M{M}.npz')
        hp_arr = np.array(hp_history) if hp_history else np.empty((0, 3))
        np.savez(
            fname,
            online_errors     = np.array(online_errors),
            online_smse       = online_smse,
            smse_val          = metrics_val['smse'],
            nlp_val           = metrics_val['nlp'],
            coverage_val      = metrics_val['coverage'],
            hp_history        = hp_arr,
            Xb_final          = model.Xb,
            lengthscale_final = model.kernel.lengthscale,
            signal_var_final  = model.kernel.variance,
            noise_var_final   = model.noise_var,
            scaler            = np.array([Y_mean, Y_std, *X_mean, *X_std]),
            num_basis         = M,
        )
        print(f"\n  结果已保存: {fname}")

    return model, metrics_val, online_smse


# ================================================================
# ===== main =====================================================
# ================================================================

def main():
    all_results = {}

    for dataset in DATASET_LIST:
        print(f"\n{'#'*65}")
        print(f"#  数据集: {dataset.upper()}")
        print(f"{'#'*65}")

        dataset_results = {}
        for i, M in enumerate(M_LIST):
            print(f"\n{'='*65}")
            print(f"  [{i+1}/{len(M_LIST)}]  dataset={dataset}  M={M}")
            print(f"{'='*65}")
            model, metrics_val, online_smse = run_single(M, dataset)
            dataset_results[M] = {
                'metrics_val': metrics_val,
                'online_smse': online_smse,
            }
        all_results[dataset] = dataset_results

    print(f"\n\n{'#'*70}")
    print(f"#  全部结果汇总")
    print(f"{'#'*70}")

    for dataset in DATASET_LIST:
        print(f"\n{'='*70}")
        print(f"  {dataset.upper()}")
        print(f"{'='*70}")
        print(f"  {'M':>4}  {'Online SMSE':>14}  "
              f"{'Val SMSE':>10}  {'NLP':>10}  {'Coverage':>10}")
        print(f"  {'-'*60}")
        for M in M_LIST:
            r  = all_results[dataset][M]
            mv = r['metrics_val']
            print(f"  {M:>4}  "
                  f"{r['online_smse']:>14.6f}  "
                  f"{mv['smse']:>10.6f}  "
                  f"{mv['nlp']:>10.6f}  "
                  f"{mv['coverage']:>10.4f}")

    save_dir = OUTPUT_CONFIG['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'rgp_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'M', 'online_smse', 'val_smse', 'nlp', 'coverage'])
        for dataset in DATASET_LIST:
            for M in M_LIST:
                r  = all_results[dataset][M]
                mv = r['metrics_val']
                writer.writerow([
                    dataset, M,
                    f"{r['online_smse']:.6f}",
                    f"{mv['smse']:.6f}",
                    f"{mv['nlp']:.6f}",
                    f"{mv['coverage']:.4f}",
                ])
    print(f"\n  汇总 CSV 已保存: {csv_path}")

    return all_results


if __name__ == '__main__':
    results = main()