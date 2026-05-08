"""
SVI-GP Runner
=============
基于 Hensman et al. (2013) Stochastic Variational Inference GP

训练阶段:
  Phase 1 (offline) — 只用 train data，全量自然梯度收敛 q(u)
                       固定：核超参数、log_beta、Xm
  Phase 2 (online)  — 顺序 mini-batch，predict-before-update
                       自然梯度更新 q(u) + Adam 更新超参数/Xm

Online data 视为未知数据原则：
  - 所有初始化（KMeans、lengthscale heuristic、noise_init、signal_var_init）
    只使用 train data 的统计量
  - N_seen 从 len(X_train) 出发，每处理一个 batch 后累加 B
  - N_seen 同步传入 update_natural_grad 和 compute_elbo，
    保证两处 scale factor 一致

N_seen 语义：
  N_seen = N_offline + N_online_seen
  表示模型当前已见数据的总规模，用于 mini-batch scale factor N_seen/B，
  控制数据项相对于 KL 正则化项的权重。
"""
import math
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import pandas as pd
import sys

# ================================================================
# ===== 配置区 =====================================
# ================================================================
# 四函数:   'rastrigin' , 'rosenbrock' , 'himmelblau' ,'sixhumpcamel'   [100, 600], [601, 2500] 
# 静态:     'boston'    , 'concrete'                                    [100, 451],[451,507];  [100, 927],[927,1030]
# 系统辨识: 'vanderpol' ,  'boucwen'   , 'tanks'    , 'building'     
#           [100, 1000]   [100:998]     [100:1023]    [100, 17520]      
#           [1000,2000]    独立vali      独立vali      [17520, 35000]

DATASET_LIST = [ 'boston'    , 'concrete'    ]
M_LIST = [30] 

# ── 训练超参（按数据集配置）──────────────────────────────────────
# phase1_epochs : Phase 1 最大 epoch 数
# lr_phase2     : Phase 2 Adam 学习率（超参数 + Xm）
# patience      : 早停等待步数（Phase 1 & Phase 2 共用）
# tol           : 早停 ELBO 提升阈值
# lr_ng         : 自然梯度步长（固定，Phase 1 & Phase 2 共用）
# batch_size    : Phase 2 mini-batch 大小
TRAIN_CONFIG_MAP = {
    # 四函数
    'rastrigin':   {'phase1_epochs': 1, 'lr_phase2': 0.005, 'patience': 30,    'tol': 1e-4, 'lr_ng': 0.01, 'batch_size': 100},
    'rosenbrock':   {'phase1_epochs': 1, 'lr_phase2': 0.005, 'patience': 30,   'tol': 1e-4, 'lr_ng': 0.01, 'batch_size': 100},
    'himmelblau':   {'phase1_epochs': 1, 'lr_phase2': 0.005, 'patience': 30,   'tol': 1e-4, 'lr_ng': 0.01, 'batch_size': 100},
    'sixhumpcamel': {'phase1_epochs': 1, 'lr_phase2': 0.005, 'patience': 30,   'tol': 1e-4, 'lr_ng': 0.01, 'batch_size': 100},
    # 静态数据集
    'boston':       {'phase1_epochs': 1, 'lr_phase2': 0.001,'patience': 40,   'tol': 1e-4, 'lr_ng': 0.005, 'batch_size': 100},
    'concrete':     {'phase1_epochs': 1, 'lr_phase2': 0.001,'patience': 40,   'tol': 1e-4, 'lr_ng': 0.005, 'batch_size': 100},
    # 系统辨识
    'vanderpol':    {'phase1_epochs': 1, 'lr_phase2': 0.002, 'patience': 50,   'tol': 1e-4, 'lr_ng': 0.01, 'batch_size': 200},
    'boucwen':      {'phase1_epochs': 1, 'lr_phase2': 0.002, 'patience': 50,   'tol': 1e-4, 'lr_ng': 0.005, 'batch_size': 200},
    'tanks':        {'phase1_epochs': 1, 'lr_phase2': 0.002, 'patience': 50,   'tol': 1e-4, 'lr_ng': 0.005, 'batch_size': 200},
    'building':     {'phase1_epochs': 1, 'lr_phase2': 0.002, 'patience': 1000,   'tol': 1e-4, 'lr_ng': 0.005, 'batch_size': 200},
}

MODEL_CONFIG = {
    'optimize_Xm': True,
    'jitter':      1e-6,
}

OUTPUT_CONFIG = {
    'save_dir':       r'D:\project\SVIGP_result',
    'save_stats':     True,
    'verbose':        True,
    'print_interval': 50,
}

FUNC_DATASETS   = {'himmelblau', 'rastrigin', 'rosenbrock', 'sixhumpcamel'}
STATIC_DATASETS = {'boston', 'concrete'}
SYSID_DATASETS  = {'vanderpol', 'boucwen', 'tanks', 'building'}

# ================================================================
# ===== 数据集类（直接复用，不改动）================================
# ================================================================
class FunctionBM:
    def __init__(self, name):
        self.paths = {
            'himmelblau':   r'd:\project\project_GP\code\MA_GP-main\data_lu\four_func\Himmelblau.csv',
            'rastrigin':    r'd:\project\project_GP\code\MA_GP-main\data_lu\four_func\Rastrigin.csv',
            'rosenbrock':   r'd:\project\project_GP\code\MA_GP-main\data_lu\four_func\Rosenbrock.csv',
            'sixhumpcamel': r'd:\project\project_GP\code\MA_GP-main\data_lu\four_func\SixHumpCamel.csv',
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
# ===== SVI-GP 模型 ===============================================
# ================================================================

class RBFKernel(nn.Module):
    """
    ARD RBF kernel: k(x,x') = σ² exp(-½ Σ_d (x_d-x'_d)²/ℓ_d²)
    参数在 log 空间存储。
    """
    def __init__(self, input_dim: int, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__()
        self.log_lengthscale = nn.Parameter(
            torch.full((input_dim,), np.log(lengthscale), dtype=torch.float64)
        )
        self.log_variance = nn.Parameter(
            torch.tensor(np.log(variance), dtype=torch.float64)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        ls  = torch.exp(self.log_lengthscale)
        var = torch.exp(self.log_variance)
        dist_sq = torch.cdist(x1 / ls, x2 / ls, p=2) ** 2
        return var * torch.exp(-0.5 * dist_sq)

    def diag(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_variance) * torch.ones(x.shape[0], dtype=torch.float64)

    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale).detach().numpy()

    @property
    def signal_variance(self):
        return torch.exp(self.log_variance).item()


class SVIGPModel(nn.Module):
    """
    Hensman et al. 2013 — L3 ELBO。

    变分分布 q(u) = N(u | m, S)，以自然参数维护：
        theta1 = S^{-1} m    (M, 1)
        theta2 = -½ S^{-1}  (M, M)
    期望参数（由 theta 推导后缓存，供预测和 loss 使用）：
        mu    = m            (M, 1)
        Sigma = S            (M, M)

    N_seen 由外部（fit 函数）在每个 batch 前传入，
    模型本身不存储也不假设数据集规模。
    """

    def __init__(
        self,
        Xm_init  : np.ndarray,
        noise_var: float = 0.1,
        lr_ng    : float = 0.01,
        jitter   : float = 1e-6,
    ):
        super().__init__()
        M, D = Xm_init.shape
        self.M      = M
        self.D      = D
        self.jitter = jitter
        self.lr_ng  = lr_ng

        # 核函数与噪声精度
        self.kernel   = RBFKernel(input_dim=D)
        self.log_beta = nn.Parameter(
            torch.tensor(np.log(1.0 / noise_var), dtype=torch.float64)
        )

        # Inducing points（Phase 2 联合优化）
        self.Xm = nn.Parameter(
            torch.tensor(Xm_init, dtype=torch.float64)
        )

        # 变分自然参数（手动管理，不走 autograd）
        # 初始化：q(u) ≈ p(u) = N(0, I)
        self.theta1 = torch.zeros((M, 1), dtype=torch.float64)
        self.theta2 = -0.5 * torch.eye(M, dtype=torch.float64)

        # 期望参数（由 theta 推导后缓存）
        self.mu    = torch.zeros((M, 1), dtype=torch.float64)
        self.Sigma = torch.eye(M,        dtype=torch.float64)

    # ── 内部工具 ─────────────────────────────────────────────────

    def _Kmm_inv(self, Xm=None):
        """返回 (Kmm, Kmm_inv, Lm)，使用 Cholesky 分解。"""
        Xm  = self.Xm if Xm is None else Xm
        Kmm = self.kernel(Xm, Xm) + self.jitter * torch.eye(self.M, dtype=torch.float64)
        Lm  = torch.linalg.cholesky(Kmm)
        return Kmm, torch.cholesky_inverse(Lm), Lm

    def _natural_to_expectation(self, theta1, theta2):
        """
        自然参数 → 期望参数：
            S = -½ theta2^{-1},   m = S theta1
        若 S 不正定返回 (None, None)。
        """
        try:
            Sigma_new = -0.5 * torch.linalg.inv(theta2)
            if torch.any(torch.linalg.eigvalsh(Sigma_new) <= 0):
                return None, None
            return Sigma_new @ theta1, Sigma_new
        except RuntimeError:
            return None, None

    # ── 自然梯度更新 q(u) ────────────────────────────────────────

    @torch.no_grad()
    def update_natural_grad(
        self,
        X_batch : np.ndarray,
        y_batch : np.ndarray,
        N_seen  : int,
    ) -> bool:
        """
        对 theta1, theta2 做一步自然梯度更新（Hensman 2013, Section 3.2）。

        自然梯度 = ∂L3/∂eta：
            ∂L3/∂eta1 = β·(N_seen/B)·Kmm^{-1} Kmn y  -  theta1
            ∂L3/∂eta2 = ½·S^{-1}  -  ½·Lambda
            Lambda    = β·(N_seen/B)·Kmm^{-1} Kmn Knm Kmm^{-1}  +  Kmm^{-1}

        N_seen 由外部传入（= N_offline + N_online_seen），
        保证 scale factor 随已见数据量动态增长。

        返回 True 表示更新成功，False 表示因数值问题跳过。
        """
        X_b = torch.as_tensor(X_batch, dtype=torch.float64)
        y_b = torch.as_tensor(y_batch, dtype=torch.float64).reshape(-1, 1)
        B     = X_b.shape[0]
        beta  = torch.exp(self.log_beta)
        scale = N_seen / B

        # 用 detach 的 Xm，与 autograd 计算图解耦
        Xm_d = self.Xm.detach()
        try:
            Kmm     = self.kernel(Xm_d, Xm_d) + self.jitter * torch.eye(self.M, dtype=torch.float64)
            Lm      = torch.linalg.cholesky(Kmm)
            Kmm_inv = torch.cholesky_inverse(Lm)
        except RuntimeError:
            return False

        Kmn = self.kernel(Xm_d, X_b)       # (M, B)
        A   = Kmm_inv @ Kmn                 # (M, B)

        # Lambda = 数据项贡献 + KL 对 q(u) 的贡献
        Lambda    = beta * scale * (A @ A.T) + Kmm_inv    # (M, M)
        data_term = beta * scale * (A @ y_b)               # (M, 1)

        # 自然梯度
        S_inv     = -2.0 * self.theta2

        grad_eta1 = data_term - self.theta1 
        grad_eta2 = 0.5 * S_inv - 0.5 * Lambda

        mu_new, Sigma_new = self._natural_to_expectation(
            self.theta1 + self.lr_ng * grad_eta1,
            self.theta2 + self.lr_ng * grad_eta2,
        )
        if mu_new is None:
            return False

        self.theta1 = self.theta1 + self.lr_ng * grad_eta1
        self.theta2 = self.theta2 + self.lr_ng * grad_eta2
        self.mu     = mu_new
        self.Sigma  = Sigma_new
        return True

    # ── L3 ELBO ──────────────────────────────────────────────────

    def compute_elbo(
        self,
        X_batch : np.ndarray,
        y_batch : np.ndarray,
        N_seen  : int,
    ) -> tuple:
        """
        L3 ELBO（Hensman 2013, eq. 4）：

            L3 = (N_seen/B) Σ_{i∈batch} {
                     log N(y_i | a_i^T m, β^{-1})
                     - ½ β k̃_{i,i}
                     - ½ tr(S Λ_i)
                 }
                 - KL( q(u) ‖ p(u) )

        N_seen 与 update_natural_grad 中的值保持一致，
        确保自然梯度和 Adam 梯度对数据规模的感知相同。

        mu 和 Sigma 视为常数（.detach()），
        梯度只流向 kernel 参数、log_beta、Xm。

        返回 (-ELBO, elbo_scalar)。
        """
        X_b = torch.as_tensor(X_batch, dtype=torch.float64)
        y_b = torch.as_tensor(y_batch, dtype=torch.float64).reshape(-1, 1)
        B     = X_b.shape[0]
        beta  = torch.exp(self.log_beta)
        scale = N_seen / B

        Kmm, Kmm_inv, Lm = self._Kmm_inv()
        Kmn      = self.kernel(self.Xm, X_b)   # (M, B)
        Knn_diag = self.kernel.diag(X_b)       # (B,)
        A        = Kmm_inv @ Kmn               # (M, B)

        mu_c    = self.mu.detach()
        Sigma_c = self.Sigma.detach()

        # 预测均值
        mean_f = (A.T @ mu_c).squeeze(1)       # (B,)

        # 残差方差 k̃_{i,i} = k(x_i,x_i) - Q_{ii}
        Qnn_diag = torch.sum(Kmn * A, dim=0)
        tilde_k  = Knn_diag - Qnn_diag

        # tr(S Λ_i) = β a_i^T S a_i
        tr_SLam = beta * torch.sum(A * (Sigma_c @ A), dim=0)

        # 逐点 log-likelihood
        residuals     = y_b.squeeze(1) - mean_f
        log_lik_terms = (
            - 0.5 * np.log(2 * np.pi)
            + 0.5 * self.log_beta
            - 0.5 * beta * residuals ** 2
            - 0.5 * beta * tilde_k
            - 0.5 * tr_SLam
        )
        sum_log_lik = scale * torch.sum(log_lik_terms)

        # KL( q(u) ‖ p(u) )  — 全局项，不含 scale
        log_det_Kmm = 2.0 * torch.sum(torch.log(torch.diagonal(Lm)))
        try:
            Ls        = torch.linalg.cholesky(Sigma_c)
            log_det_S = 2.0 * torch.sum(torch.log(torch.diagonal(Ls)))
        except RuntimeError:
            log_det_S = torch.logdet(Sigma_c)

        kl = 0.5 * (
            torch.trace(Kmm_inv @ Sigma_c)
            + (mu_c.T @ Kmm_inv @ mu_c).squeeze()
            - self.M
            + log_det_Kmm - log_det_S
        )

        elbo = sum_log_lik - kl
        return -elbo, elbo.detach().item()

    # ── 预测 ─────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, X_star: np.ndarray) -> tuple:
        """
        后验预测  p(y* | x*) ≈ N(μ*, σ²_y*)

        μ*    = K_{*m} Kmm^{-1} m
        σ²_f* = k(x*,x*) - diag(Q_{**})
                + diag(K_{*m} Kmm^{-1} S Kmm^{-1} K_{m*})
        σ²_y* = σ²_f* + β^{-1}

        返回 (mean (n*,), var (n*,))，numpy 数组。
        """
        X_s = torch.as_tensor(X_star, dtype=torch.float64)
        _, Kmm_inv, _ = self._Kmm_inv()
        Kms      = self.kernel(self.Xm, X_s)
        Kss_diag = self.kernel.diag(X_s)
        A        = Kmm_inv @ Kms

        mean  = (A.T @ self.mu).squeeze(1)
        var_f = (
            Kss_diag
            - torch.sum(Kms * A, dim=0)
            + torch.sum((self.Sigma @ A) * A, dim=0)
        )
        var_y = torch.clamp(var_f + 1.0 / torch.exp(self.log_beta), min=1e-8)
        return mean.numpy(), var_y.numpy()

    # ── 参数分组 ─────────────────────────────────────────────────

    def get_hyperparams(self) -> list:
        """Phase 2 Adam：核超参数 + log_beta + Xm。"""
        return list(self.kernel.parameters()) + [self.log_beta, self.Xm]

    def get_hyperparams_no_Xm(self) -> list:
        """备用：不含 Xm。"""
        return list(self.kernel.parameters()) + [self.log_beta]

    @torch.no_grad()
    def get_hyperparameters(self) -> dict:
        return {
            'lengthscale': self.kernel.lengthscale,
            'signal_var':  self.kernel.signal_variance,
            'noise_var':   (1.0 / torch.exp(self.log_beta)).item(),
        }


# ================================================================
# ===== 训练入口 ==================================================
# ================================================================

def fit(
    model          : SVIGPModel,
    X_train        : np.ndarray,
    y_train        : np.ndarray,
    X_online       : np.ndarray,
    y_online       : np.ndarray,
    cfg            : dict,
    N_offline      : int,
    Y_mean         : float,
    Y_std          : float,
    verbose        : bool = True,
    print_interval : int  = 50,
):
    """
    两阶段训练。

    Phase 1（offline）：
      - 每 epoch 用 X_train 全量做一次自然梯度更新
      - N_seen = N_offline（固定，offline 阶段不增长）
      - 更新后 no_grad 计算 train ELBO 用于早停
      - 超参数、Xm 固定不动

    Phase 2（online）：
      - 顺序遍历 online data，每个 mini-batch：
          1. predict（标准化尺度 → 反标准化，记录原始误差）
          2. update_natural_grad（N_seen 传入当前值）
          3. compute_elbo → Adam 更新超参数 + Xm（N_seen 传入当前值）
          4. N_seen += B
      - 监控 ELBO 早停

    N_seen 初始值 = N_offline，每处理一个 online batch 后累加 B，
    保证 update_natural_grad 和 compute_elbo 使用完全相同的 scale factor。

    返回:
        elbo_phase1  : list[float]，Phase 1 每 epoch 的 train ELBO
        elbo_phase2  : list[float]，Phase 2 每 step 的 mini-batch ELBO
        online_errors: list[float]，Phase 2 每 batch 的原始尺度 MSE
                       （predict-before-update，用于汇总 online SMSE）
    """
    elbo_phase1   = []
    elbo_phase2   = []
    online_errors = []

    patience = cfg['patience']
    tol      = cfg['tol']

    # ── Phase 1：offline，自然梯度收敛 q(u) ──────────────────────
    if verbose:
        print("=" * 65)
        print("Phase 1 (offline): Natural gradient  "
              "(train data 全量，超参数/Xm 固定)")
        print("=" * 65)

    best_elbo_p1 = -np.inf
    wait_p1      = 0

    for ep in range(cfg['phase1_epochs']):
        # 全量 train，N_seen = N_offline（offline 阶段固定）
        ok = model.update_natural_grad(X_train, y_train, N_seen=N_offline)

        # no_grad 前向计算 train ELBO，用于早停
        with torch.no_grad():
            _, elbo_val = model.compute_elbo(X_train, y_train, N_seen=N_offline)
        elbo_phase1.append(elbo_val)

        if elbo_val > best_elbo_p1 + tol:
            best_elbo_p1 = elbo_val
            wait_p1      = 0
        else:
            wait_p1 += 1
            if wait_p1 >= patience:
                if verbose:
                    print(f"  Phase 1 早停 @ epoch {ep+1}  "
                          f"(ELBO={elbo_val:.4f})")
                break

    if verbose:
        print(f"  Phase 1 结束  |  epochs={len(elbo_phase1)}  "
              f"|  最终 train ELBO={elbo_phase1[-1]:.4f}")
        # ── 打印超参数（需要时取消注释）──────────────────────────
        # hp = model.get_hyperparameters()
        # print(f"    ℓ={hp['lengthscale'].round(4)}  "
        #       f"σ²={hp['signal_var']:.4f}  noise={hp['noise_var']:.6f}")

    # ── Phase 2：online，联合优化 ─────────────────────────────────
    if verbose:
        print()
        print("=" * 65)
        print("Phase 2 (online): Joint optimization  "
              "(顺序 mini-batch，predict-before-update)")
        print("=" * 65)

    optimizer    = optim.Adam(model.get_hyperparams(), lr=cfg['lr_phase2'])
    batch_size   = cfg['batch_size']
    N_seen       = N_offline
    step         = 0
    start        = 0

    while True:
        Xb_np = X_online[start : start + batch_size]
        if len(Xb_np) == 0:
            break
        yb_np = y_online[start : start + batch_size]
        B     = len(Xb_np)
        # 步骤 1：predict-before-update（原始尺度）
        mean_s, _ = model.predict(Xb_np)
        mean_orig = mean_s * Y_std + Y_mean
        y_orig    = yb_np * Y_std + Y_mean
        online_errors.append(float(np.mean((mean_orig - y_orig) ** 2)))

        # 步骤 2：自然梯度更新 q(u)（使用 N_seen，更新前的已见数据量）
        model.update_natural_grad(Xb_np, yb_np, N_seen=N_seen)

        # 步骤 3：Adam 更新超参数 + Xm（同一 N_seen）
        optimizer.zero_grad()
        loss, elbo_val = model.compute_elbo(Xb_np, yb_np, N_seen=N_seen)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_hyperparams(), max_norm=5.0)
        optimizer.step()

        # 步骤 4：本 batch 处理完，累加 N_seen
        N_seen += B

        elbo_phase2.append(elbo_val)
        start += batch_size
        step += 1

        # 打印
        if verbose and step % print_interval == 0:
            y_seen_orig = y_online[:start] * Y_std + Y_mean
            online_smse_now = (
                float(np.mean(online_errors) / np.var(y_seen_orig))
                if len(y_seen_orig) > 1 else float('nan')
                )
            print(f"  [step {step:4d}]  "
                  f"ELBO={elbo_val:10.3f}  |  "
                  f"N_seen={N_seen}  |  "
                  f"online SMSE(so far)={online_smse_now:.6f}")
            # ── 打印超参数（需要时取消注释）──────────────────────
            # hp = model.get_hyperparameters()        
            # print(f"    ℓ={hp['lengthscale'].round(4)}  "
            #         f"σ²={hp['signal_var']:.4f}  noise={hp['noise_var']:.6f}")

    if verbose:
        print(f"  Phase 2 结束  |  steps={step}  "
              f"|  最终 mini-batch ELBO={elbo_phase2[-1]:.4f}  "
              f"|  最终 N_seen={N_seen}")

    return elbo_phase1, elbo_phase2, online_errors


# ================================================================
# ===== 评估指标 ==================================================
# ================================================================

def compute_metrics(mean, var, y_true):
    """
    SMSE、NLP、Coverage（95% CI），输入均为原始尺度。
    SMSE < 1 表示优于均值基线。
    """
    err2     = (mean - y_true) ** 2
    smse     = float(err2.mean() / np.var(y_true))
    nlp      = float(np.mean(0.5 * err2 / var + 0.5 * np.log(2 * np.pi * var)))
    std      = np.sqrt(var)
    coverage = float(((y_true >= mean - 1.96*std) & (y_true <= mean + 1.96*std)).mean())
    return {'smse': smse, 'nlp': nlp, 'coverage': coverage}


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
# ===== 标准化（与 VFE runner 逻辑一致，只用 train 统计量）=========
# ================================================================

def build_scalers(X_train, y_train, dataset=''):
    col_std  = X_train.std(axis=0)
    col_mean = X_train.mean(axis=0)

    # building 的 X 和 y 量纲特殊，强制全列标准化
    #if dataset.lower() == 'building':
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


def median_heuristic(X):
    """只应传入 train data 的标准化结果。"""
    dists = pairwise_distances(X).flatten()
    return float(np.median(dists[dists > 0]))


# ================================================================
# ===== 单次训练 + 评估 ==========================================
# ================================================================

def run_single(M, dataset):
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"\n{'='*65}")
    print(f"  SVI-GP  |  数据集: {dataset.upper()}  |  M={M}")
    print(f"{'='*65}")

    # ── 1. 加载数据 ──────────────────────────────────────────────
    X_tr, y_tr, X_on, y_on, X_val, y_val, info = load_dataset(dataset)

    # ── 2. 标准化（只用 train 统计量，online data 视为未知）───────
    X_mean, X_std, Y_mean, Y_std = build_scalers(X_tr, y_tr,dataset=dataset)

    X_tr_s  = scale_X(X_tr,  X_mean, X_std)
    X_on_s  = scale_X(X_on,  X_mean, X_std)
    X_val_s = scale_X(X_val, X_mean, X_std)
    y_tr_s  = scale_y(y_tr,  Y_mean, Y_std)
    y_on_s  = scale_y(y_on,  Y_mean, Y_std)

    # ── 3. 打印数据信息 ──────────────────────────────────────────
    print(f"\n  train:    {info['N_train']} 条  |  "
          f"online: {info['N_online']} 条  |  "
          f"validate: {info['N_test']} 条  |  D={info['D']}")
    print("  train y:    mean={:.2f}, std={:.2f}, "
          "min={:.2f}, max={:.2f}".format(
          y_tr.mean(), y_tr.std(), y_tr.min(), y_tr.max()))
    print("  online y:   mean={:.2f}, std={:.2f}, "
          "min={:.2f}, max={:.2f}".format(
          y_on.mean(), y_on.std(), y_on.min(), y_on.max()))
    print("  validate y: mean={:.2f}, std={:.2f}, "
          "min={:.2f}, max={:.2f}".format(
          y_val.mean(), y_val.std(), y_val.min(), y_val.max()))
    print(f"y_tr_s stats: mean={y_tr_s.mean():.4f}, std={y_tr_s.std():.4f}, "
        f"min={y_tr_s.min():.4f}, max={y_tr_s.max():.4f}")
    print(f"Y_mean={Y_mean:.4f}, Y_std={Y_std:.4f}")
    print(f"X_tr_s 各列 std: {X_tr_s.std(axis=0).round(4)}")
    print(f"X_mean={X_mean.round(4)}")
    print(f"X_std={X_std.round(4)}")
    # ── 4. 初始化（只用 train data）──────────────────────────────
    ls_init    = median_heuristic(X_tr_s)          # 只用 train
    y_std_s    = float(y_tr_s.std())               # 只用 train
    noise_init = max(y_std_s * 0.1, 1e-4)

    km      = KMeans(n_clusters=M, random_state=42, n_init=10).fit(X_tr_s)  # 只用 train
    Xm_init = km.cluster_centers_.astype(np.float64)

    cfg = TRAIN_CONFIG_MAP[dataset.lower()]

    model = SVIGPModel(
        Xm_init=Xm_init,
        noise_var=noise_init,
        lr_ng=cfg['lr_ng'],
        jitter=MODEL_CONFIG['jitter'],
    )
    with torch.no_grad():
        model.kernel.log_lengthscale.fill_(np.log(ls_init))
        model.kernel.log_variance.fill_(np.log(max(y_std_s ** 2, 1e-4)))

    N_offline = len(X_tr_s)   # Phase 1 的 N_seen；Phase 2 的 N_seen 初始值

    print(f"\n  初始超参: ℓ={ls_init:.4f}, "
          f"σ²={y_std_s**2:.4f}, noise={noise_init:.5f}")
    print(f"  模型配置: M={M}, optimize_Xm={MODEL_CONFIG['optimize_Xm']}, "
          f"lr_ng={cfg['lr_ng']}")
    print(f"  N_offline={N_offline}  "
          f"(Phase 2 N_seen 从此出发，随 online batch 累加)")

    # ── 5. 训练 ──────────────────────────────────────────────────
    elbo_p1, elbo_p2, online_errors = fit(
        model=model,
        X_train=X_tr_s,   y_train=y_tr_s,
        X_online=X_on_s,  y_online=y_on_s,
        cfg=cfg,
        N_offline=N_offline,
        Y_mean=Y_mean,     Y_std=Y_std,
        verbose=OUTPUT_CONFIG['verbose'],
        print_interval=OUTPUT_CONFIG['print_interval'],
    )

    # ── 6. 评估 ──────────────────────────────────────────────────
    # Online SMSE（predict-before-update，原始尺度）
    online_smse = (
        float(np.mean(online_errors) / np.var(y_on))
        if len(online_errors) > 0 else float('nan')
    )

    # Validation 评估
    mean_s, var_s = model.predict(X_val_s)
    mean_orig     = unscale_mean(mean_s, Y_mean, Y_std)
    var_orig      = unscale_var(var_s,   Y_std)
    metrics_val   = compute_metrics(mean_orig, var_orig, y_val)

    # ── 7. 打印结果 ───────────────────────────────────────────────
    # print(f"\n{'='*65}")
    # print(f"  最终结果 — {dataset.upper()}  M={M}")
    # print(f"{'='*65}")
    # print(f"  Online  SMSE (predict-before-update): {online_smse:.6f}")
    # print(f"  Validate SMSE:     {metrics_val['smse']:.6f}  （< 1 优于均值基线）")
    # print(f"  Validate NLP:      {metrics_val['nlp']:.6f}")
    # print(f"  Validate Coverage: {metrics_val['coverage']:.4f}  （目标 ≈ 0.95）")
    # ── 打印收敛超参数（需要时取消注释）────────────────────────────
    print(f"\n  收敛超参:")
    for k, v in model.get_hyperparameters().items():
        print(f"    {k:15s}: {v}")

    # ── 8. 保存结果 ───────────────────────────────────────────────
    if OUTPUT_CONFIG['save_stats']:
        save_dir = OUTPUT_CONFIG['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f'svigp_{dataset}_M{M}.npz')
        np.savez(
            fname,
            elbo_phase1   = np.array(elbo_p1),
            elbo_phase2   = np.array(elbo_p2),
            online_errors = np.array(online_errors),
            online_smse   = online_smse,
            smse_val      = metrics_val['smse'],
            nlp_val       = metrics_val['nlp'],
            coverage_val  = metrics_val['coverage'],
            Xm_final      = model.Xm.detach().numpy(),
            lengthscale   = model.get_hyperparameters()['lengthscale'],
            signal_var    = model.get_hyperparameters()['signal_var'],
            noise_var     = model.get_hyperparameters()['noise_var'],
            scaler        = np.array([Y_mean, Y_std, *X_mean, *X_std]),
            N_offline     = N_offline,
            num_Xm        = M,
        )
        print(f"\n  结果已保存: {fname}")

    return model, metrics_val, online_smse, elbo_p1, elbo_p2


# ================================================================
# ===== 主入口 ====================================================
# ================================================================

def main():
    all_results = {}

    # ── 训练阶段：只跑，不打印汇总 ───────────────────────────────
    for dataset in DATASET_LIST:
        print(f"\n{'#'*65}")
        print(f"#  数据集: {dataset.upper()}")
        print(f"{'#'*65}")

        dataset_results = {}
        for i, M in enumerate(M_LIST):
            print(f"\n{'='*65}")
            print(f"  [{i+1}/{len(M_LIST)}]  dataset={dataset}  M={M}")
            print(f"{'='*65}")
            model, metrics_val, online_smse, elbo_p1, elbo_p2 = run_single(M, dataset)
            dataset_results[M] = {
                'metrics_val': metrics_val,
                'online_smse': online_smse,
            }

        all_results[dataset] = dataset_results

    # ── 所有数据集训练完毕，统一打印 ─────────────────────────────
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
    csv_path = os.path.join(save_dir, f'svigp_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'M', 'online_smse', 'val_smse', 'nlp', 'coverage'])
        for dataset in DATASET_LIST:
            for M in M_LIST:
                r  = all_results[dataset][M]
                mv = r['metrics_val']
                writer.writerow([
                    dataset,
                    M,
                    f"{r['online_smse']:.6f}",
                    f"{mv['smse']:.6f}",
                    f"{mv['nlp']:.6f}",
                    f"{mv['coverage']:.4f}",
                ])

    print(f"\n汇总 CSV 已保存: {csv_path}")
       



    return all_results


if __name__ == '__main__':
    results = main()