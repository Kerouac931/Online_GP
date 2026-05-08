"""
SRGP Runner — 集成版
=====================
数据集: Himmelblau / Rastrigin / Rosenbrock / SixHumpCamel
        Boston / Concrete
        VanDerPol / Building / BoucWen / Tanks

数据划分逻辑（与原代码一致）:
  train    → SRGP 初始训练集（前100条）
  online   → SRGP 流式更新数据
  validate → 测试集（评估用）
"""

import os
import numpy as np
import pandas as pd
import GPy
from sklearn.cluster import KMeans
import sys
import csv
from RECC import REC
from optim import Adam

# ================================================================
# ===== 配置区 =====================================
# ================================================================
# 四函数:   'rastrigin' , 'rosenbrock' , 'himmelblau' ,'sixhumpcamel'  [100, 600], [601, 2500] 
# 静态:     'boston'    , 'concrete'                                    [100, 451],[451,507];  [100, 927],[927,1030]
# 系统辨识: 'vanderpol' ,  'boucwen'   , 'tanks'    , 'building'     
#           [100, 1000]   [100:998]     [100:1023]    [100, 17520]      
#           [1000,2000]    独立vali      独立vali      [17520, 35000]

DATASET_LIST = ['building']
M_LIST = [50]

online_batch = 200
offline_batch = 100 

TRAIN_CONFIG = {
    'online_update':      True,
    'n_epochs':           1,
    'batch_size':         offline_batch,
    'batch_size_online':  online_batch,
    'lr':                 0.001,
    'permute':            True,
    'estimate': {'σ0': True, 'ls': True, 'σn': True, 'R': True},
}

MODEL_CONFIG = {
    'M':      0, # replaced by M_LIST
    'alpha':  0.5,
    #'alpha':  0.0,
    'kernel': 'RBF',
    'ARD':    True,
}

OUTPUT_CONFIG = {
    'save_stats': True,
    'plot':       False,
    'plot_y':     True,
}

# ================================================================
# ===== 数据加载（Class 不改动）==================================
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
            train_index = 1000
            result['train']    = data.iloc[: train_index]
            result['online']   = data.iloc[train_index: 17520]
            result['validate'] = data.iloc[17520:]
        elif self.name in ['boucwen', 'tanks']:
            result['online'] = data.iloc[100:]
            if self.name in self.validate_file_map:
                result['validate'] = self._read_csv(self.validate_file_map[self.name])
            else:
                result['validate'] = pd.DataFrame()
        return result


# ── 数据集路由 ───────────────────────────────────────────────────
FUNC_DATASETS   = {'himmelblau', 'rastrigin', 'rosenbrock', 'sixhumpcamel'}
STATIC_DATASETS = {'boston', 'concrete'}
SYSID_DATASETS  = {'vanderpol', 'boucwen', 'tanks', 'building'}


def to_Xy(split_data):
    """DataFrame 或 dict 统一转 numpy float，y 为 1D"""
    if isinstance(split_data, pd.DataFrame):
        if split_data.empty:
            return np.empty((0, 0)), np.empty(0)
        X = split_data.iloc[:, :-1].values.astype(float)
        y = split_data.iloc[:, -1].values.astype(float)
    else:
        X = np.array(split_data['X'], dtype=float)
        y = np.array(split_data['y'], dtype=float).ravel()
    return X, y


def load_dataset(name):
    """
    返回六个数组，y 均为 1D：
        X_tr  (N_tr, D)   y_tr  (N_tr,)   ← 初始训练集
        X_on  (N_on, D)   y_on  (N_on,)   ← 在线更新集
        X_val (N_te, D)   y_val (N_te,)   ← 验证集
    """
    name = name.lower()
    if   name in FUNC_DATASETS:   splits = FunctionBM(name).get_splits()
    elif name in STATIC_DATASETS: splits = StaticDatasetBM(name).get_splits()
    elif name in SYSID_DATASETS:  splits = SysIDBM(name).get_splits()
    else: raise ValueError(f"未知数据集: {name}")

    X_tr,  y_tr  = to_Xy(splits['train'])
    X_on,  y_on  = to_Xy(splits['online'])
    X_val, y_val = to_Xy(splits['validate'])

    return X_tr, y_tr, X_on, y_on, X_val, y_val


# ================================================================
# ===== 标准化工具函数 ============================================
# ================================================================

def build_scalers(X_train, y_train, dataset=''):
    """
    只用 train split 的统计量拟合 scaler，online data 对此不可见。
    X_train: (N, D)  y_train: (N,) 1D
    """
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


def scale_X(X, X_mean, X_std):       return (X - X_mean) / X_std
def scale_y(y, Y_mean, Y_std):       return (y - Y_mean) / Y_std
def unscale_mean(m, Y_mean, Y_std):  return m * Y_std + Y_mean
def unscale_var(v, Y_std):           return v * (Y_std ** 2)

def median_heuristic(X):
    """用训练集输入的成对距离中位数估计 lengthscale"""
    from scipy.spatial.distance import pdist
    dists = pdist(X, metric='euclidean')
    return float(np.median(dists)) if len(dists) > 0 else 1.0

def save_results_csv(results, save_path=r'D:\project\SRGP_result\results.csv'):
    """追加写入，文件不存在时自动创建表头"""
    fieldnames = [
        'algorithm', 'dataset', 'num_Xm',
        'online_SMSE', 'val_SMSE',
        'online_batch', 'online_update',
        'lr'
    ]
    file_exists = os.path.isfile(save_path)
    with open(save_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for (dataset, M), res in results.items():
            writer.writerow({
                'algorithm':    'SRGP',
                'dataset':      dataset.upper(),
                'num_Xm':       M,
                'online_SMSE':  f"{res['smse_online']:.6f}",
                'val_SMSE':     f"{res['smse_val']:.6f}",
                'online_batch': TRAIN_CONFIG['batch_size_online'],
                'online_update': TRAIN_CONFIG['online_update'],
                'lr':            TRAIN_CONFIG['lr'],
            })
    print(f"结果已追加写入: {save_path}")

# ================================================================
# ===== 模型初始化（不改动）======================================
# ================================================================

def init_params(X_train, Y_train, mc, dataset=''):
    N, D = X_train.shape
    kern_cls = {
        'RBF':      GPy.kern.RBF,
        'Matern32': GPy.kern.Matern32,
        'Matern52': GPy.kern.Matern52,
    }[mc['kernel']]

    if dataset == 'building':
        ls_init = median_heuristic(X_train) # input = X_train_s when called
    else:
        ls_init = median_heuristic(X_train)
    y_std        = float(np.std(Y_train))
    sigma0_init  = max(y_std, 1e-3)
    sigma_n_init = max(y_std * 0.1, 1e-4)

    num_ls = D if mc['ARD'] else 1

    kern = kern_cls(input_dim=D, lengthscale=ls_init, ARD=mc['ARD'])
    kern.variance = sigma0_init ** 2

    kmeans = KMeans(n_clusters=mc['M'], random_state=42, n_init=10)
    kmeans.fit(X_train)

    params = {
        'σ0': sigma0_init,
        'ls': np.full(num_ls, ls_init),
        'σn': sigma_n_init,
        'R':  kmeans.cluster_centers_,
    }
    return kern, params


def build_optimizers(params, lr):
    M, D = params['R'].shape
    return {
        'σ0': Adam(lr, (1,)),
        'ls': Adam(lr, (len(params['ls']),)),
        'σn': Adam(lr, (1,)),
        'R':  Adam(lr, (M, D)),
    }


def plot_results(stats, dataset):
    try:
        import matplotlib.pyplot as plt
        epochs = np.arange(stats.shape[1])
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'SRGP -- {dataset}', fontsize=13)
        axes[0].plot(epochs, stats[0], 'b-o', ms=3); axes[0].set_title('Log Marginal Likelihood')
        axes[1].plot(epochs, stats[1], 'r-o', ms=3); axes[1].set_title('RMSE (validate)')
        axes[2].plot(epochs, stats[3], 'g-o', ms=3)
        axes[2].axhline(y=0.95, color='k', ls='--', alpha=0.5)
        axes[2].set_title('Coverage (95% CI)'); axes[2].set_ylim([0, 1.05])
        for ax in axes:
            ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f'srgp_{dataset}_curves.png'
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        print(f"训练曲线已保存: {fname}")
        plt.show()
    except Exception as e:
        print(f"绘图跳过: {e}")

def plot_y_results(y_pred_online, y_true_online,
                   y_pred_val,    y_true_val,
                   dataset, M):
    try:
        import matplotlib.pyplot as plt

        n_on  = len(y_true_online)
        n_val = len(y_true_val)

        # validation 的 x 轴紧接 online 之后
        t_on  = np.arange(n_on)
        t_val = np.arange(n_on, n_on + n_val)

        fig, ax = plt.subplots(figsize=(16, 5))
        fig.suptitle(f'SRGP — {dataset.upper()}  M={M}  预测 vs 真实', fontsize=13)

        # 真实值
        ax.plot(t_on,  y_true_online, 'k-', lw=1.0, label='y_true (online)')
        ax.plot(t_val, y_true_val,    'k-', lw=1.0, label='y_true (val)')

        # 预测值
        ax.plot(t_on,  y_pred_online, 'r--', lw=1.0, label='y_pred (online)', alpha=0.8)
        ax.plot(t_val, y_pred_val,    'b--', lw=1.0, label='y_pred (val)',    alpha=0.8)

        # online / val 分界线
        ax.axvline(x=n_on, color='gray', ls=':', lw=1.2, label=f'online|val split (x={n_on})')

        ax.set_xlabel('Sample index')
        ax.set_ylabel('y')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'srgp_{dataset}_M{M}_y_pred.png'
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        print(f"预测曲线已保存: {fname}")
        plt.show()
    except Exception as e:
        print(f"plot_y 跳过: {e}")


# ================================================================
# ===== 单次训练 ==================================================
# ================================================================

def run_single(M, dataset):
    np.random.seed(42)
    mc = {**MODEL_CONFIG, 'M': M}
    print("=" * 60)
    print(f"  SRGP  |  数据集: {dataset.upper()}  |  M={M}")
    print("=" * 60)

    # ── 1. 加载数据（三段分开，y 均为 1D）───────────────────────
    X_tr, y_tr, X_on, y_on, X_val, y_val = load_dataset(dataset)
    print(f"\n各段 y 分布:")
    print(f"  train:    mean={y_tr.mean():.4f}, std={y_tr.std():.4f}, min={y_tr.min():.4f}, max={y_tr.max():.4f}")
    print(f"  online:   mean={y_on.mean():.4f}, std={y_on.std():.4f}, min={y_on.min():.4f}, max={y_on.max():.4f}")
    print(f"  validate: mean={y_val.mean():.4f}, std={y_val.std():.4f}, min={y_val.min():.4f}, max={y_val.max():.4f}")


    # ── 2. 标准化：只用 train 统计量，online/val 对此不可见 ──────
    X_mean, X_std, Y_mean, Y_std = build_scalers(X_tr, y_tr, dataset=dataset)

    X_tr_s  = scale_X(X_tr,  X_mean, X_std)
    X_on_s  = scale_X(X_on,  X_mean, X_std)
    X_val_s = scale_X(X_val, X_mean, X_std)
    y_tr_s  = scale_y(y_tr,  Y_mean, Y_std)
    y_on_s  = scale_y(y_on,  Y_mean, Y_std)
    y_val_s = scale_y(y_val, Y_mean, Y_std)

    print(f"\n标准化后 y 分布:")
    print(f"  y_tr_s:  mean={y_tr_s.mean():.4f}, std={y_tr_s.std():.4f}, min={y_tr_s.min():.4f}, max={y_tr_s.max():.4f}")
    print(f"  y_val_s: mean={y_val_s.mean():.4f}, std={y_val_s.std():.4f}, min={y_val_s.min():.4f}, max={y_val_s.max():.4f}")
    print(f"  Y_mean={Y_mean:.4f}, Y_std={Y_std:.4f}")
    print(f"X_tr_s  各列范围: min={X_tr_s.min(axis=0).round(2)}, max={X_tr_s.max(axis=0).round(2)}")
    print(f"X_val_s 各列范围: min={X_val_s.min(axis=0).round(2)}, max={X_val_s.max(axis=0).round(2)}")
    print(f"X_on_s  各列范围: min={X_on_s.min(axis=0).round(2)},  max={X_on_s.max(axis=0).round(2)}")
    # ── 3. 打印数据信息 ──────────────────────────────────────────
    print(f"\n数据划分:")
    print(f"  初始训练 (train):   {len(X_tr)} 条")
    print(f"  在线更新 (online):  {len(X_on)} 条")
    print(f"  ─────────────────────────────")
    print(f"  验证集 (validate): {len(X_val)} 条  |  D={X_tr.shape[1]}")

    # ── 4. 初始化模型参数和优化器（只用 train 数据）─────────────
    kern, params = init_params(X_tr_s, y_tr_s, mc, dataset)
    params_opt   = build_optimizers(params, TRAIN_CONFIG['lr'])

    print(f"\n初始超参: σs={params['σ0']:.4f}, ls={params['ls']}, σn={params['σn']:.4f}")
    print(f"模型配置: M={mc['M']}, α={mc['alpha']}, kernel={mc['kernel']}")
    print(f"训练配置: epochs={TRAIN_CONFIG['n_epochs']}, "
          f"batch={TRAIN_CONFIG['batch_size']}, lr={TRAIN_CONFIG['lr']}\n")

    # ── 5. 只用 train 数据训练初始模型 ───────────────────────────
    # REC 要求 Y 为 2D (N, 1)，Ftest/Ytest 为 1D
    model_init = REC(
        X          = X_tr_s,
        Y          = y_tr_s[:, None],
        kernel     = kern,
        nEpochs    = TRAIN_CONFIG['n_epochs'],
        batchsize  = TRAIN_CONFIG['batch_size'],
        params     = params,
        params_OPT = params_opt,
        params_EST = TRAIN_CONFIG['estimate'],
        α          = mc['alpha'],
        PERM       = TRAIN_CONFIG['permute'],
        Xtest      = X_val_s,
        Ftest      = y_val_s,
        Ytest      = y_val_s,
    )
    model_init.run()
    print(f"\n各 epoch RMSE (scaled space):")
    print(np.round(model_init.STATS[1], 4))
    print(f"各 epoch LML:")
    print(np.round(model_init.STATS[0], 4))
    print(f"各 epoch 超参 σ0:")
    print(np.round(model_init.STATS_Y[0], 6))
    print(f"各 epoch 超参 ls:")
    print(np.round(model_init.STATS_Y[1], 6))
    print(f"各 epoch 超参 σn:")
    print(np.round(model_init.STATS_Y[-1], 6))


    # ── 6. Online phase: predict-then-update ─────────────────────
    batch_size_online = TRAIN_CONFIG['batch_size_online']
    n_online          = len(X_on_s)

    online_errors        = []
    online_rmse_curve    = []
    online_smse_curve    = []
    online_ls_curve      = []
    online_sigma0_curve  = []
    online_sigma_n_curve = []
    y_pred_online_list = []
    y_true_online_list = []

    for start in range(0, n_online, batch_size_online):
        end     = min(start + batch_size_online, n_online)
        X_batch = X_on_s[start:end]
        Y_batch = y_on_s[start:end]           # 1D，inference 需要 2D

        # 先预测（原始尺度）
        m_pred, _ = model_init.predict_diag(X_batch, NOISE=False)
        m_orig    = unscale_mean(m_pred, Y_mean, Y_std)
        y_orig    = unscale_mean(Y_batch, Y_mean, Y_std)
        errors    = (m_orig - y_orig) ** 2
        y_pred_online_list.extend(m_orig.tolist())
        y_true_online_list.extend(y_orig.tolist())
        online_errors.extend(errors)


        # 记录指标
        online_rmse_curve.append(float(np.sqrt(errors.mean())))
        online_smse_curve.append(float(np.array(online_errors).mean() / np.var(y_on)))
        online_ls_curve.append(model_init.params['ls'].copy())
        online_sigma0_curve.append(float(model_init.params['σ0']) ** 2)
        online_sigma_n_curve.append(float(model_init.params['σn']) ** 2)

        # 再更新（inference 需要 Y 为 2D）
        model_init._log_marginal_likelihood, \
        model_init.n,        model_init.m,      \
        model_init.C,        model_init.P,      \
        model_init._log_Det_C,                  \
        model_init.dn_dR,    model_init.dC_dR,  model_init.dψ_dR,   \
        model_init.dn_dσ02,  model_init.dC_dσ02, model_init.dψ_dσ02, \
        model_init.dn_dl,    model_init.dC_dl,  model_init.dψ_dl,   \
        model_init.dn_dσn2,  model_init.dC_dσn2, model_init.dψ_dσn2 \
            = model_init.inference(
                model_init.n,        model_init.C,        model_init.P,
                model_init._log_marginal_likelihood,
                model_init._log_Det_C,
                model_init.dn_dR,    model_init.dC_dR,    model_init.dψ_dR,
                model_init.dn_dσ02,  model_init.dC_dσ02,  model_init.dψ_dσ02,
                model_init.dn_dl,    model_init.dC_dl,    model_init.dψ_dl,
                model_init.dn_dσn2,  model_init.dC_dσn2,  model_init.dψ_dσn2,
                X_batch, Y_batch[:, None],     # inference 需要 (N, 1)
            )

        if TRAIN_CONFIG['online_update']:
            model_init.update_params()  
            '''
            inverse(Krr) = iKrr, 与inducingpoints位置ls,signal variance 三者有关
            在update params之后马上更新,在下一次online predict时就可以在节拍上同步,不然会慢一个节拍
            '''
            if model_init.params_EST.get('R'):
                from GPy.util import diag as gpy_diag
                from GPy.util.linalg import jitchol, dpotri
                Z = model_init.params['R']
                Krr = model_init.kern.K(Z)
                gpy_diag.add(Krr, model_init.const_jitter)
                L = jitchol(Krr)
                model_init.iKrr, _ = dpotri(L, lower=1)    


    print(f"online 结束后:")
    print(f"  norm(n):    {np.linalg.norm(model_init.n):.4f}")
    print(f"  norm(C):    {np.linalg.norm(model_init.C):.4f}")
    print(f"  norm(P):    {np.linalg.norm(model_init.P):.6f}")
    print(f"  norm(m):    {np.linalg.norm(model_init.m):.6f}")
    print(f"offline 结束后 norm(m): {np.linalg.norm(model_init.m):.6f}")
    # ── 7. 计算最终 SMSE ─────────────────────────────────────────
    online_errors = np.array(online_errors)
    smse_online   = online_errors.mean() / np.var(y_on)

    m_val, _   = model_init.predict_diag(X_val_s, NOISE=False)
    m_val_orig = unscale_mean(m_val,  Y_mean, Y_std)
    smse_val   = np.mean((m_val_orig - y_val) ** 2) / np.var(y_val)
    y_pred_val_list = m_val_orig.tolist()
    y_true_val_list = y_val.tolist()

    # ── 8. 保存结果 ───────────────────────────────────────────────
    if OUTPUT_CONFIG['save_stats']:
        save_dir = r'D:\project\SRGP_result'
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f'srgp_{dataset}_M{M}.npz')
        np.savez(
            fname,
            stats                = model_init.STATS,
            stats_hyp            = model_init.STATS_Y,
            stats_R              = model_init.STATS_RR,
            R_final              = model_init.params['R'],
            sigma0               = model_init.params['σ0'],
            ls                   = model_init.params['ls'],
            sigma_n              = model_init.params['σn'],
            smse_online          = smse_online,
            smse_val             = smse_val,
            scaler               = np.array([Y_mean, Y_std, *X_mean, *X_std]),
            num_Xm               = M,
            n_epochs             = TRAIN_CONFIG['n_epochs'],
            online_update        = int(TRAIN_CONFIG['online_update']),
            online_rmse_curve    = np.array(online_rmse_curve),
            online_smse_curve    = np.array(online_smse_curve),
            online_sigma0_curve  = np.array(online_sigma0_curve),
            online_sigma_n_curve = np.array(online_sigma_n_curve),
            online_ls_curve      = np.array(online_ls_curve),
            y_pred_online = np.array(y_pred_online_list),
            y_true_online = np.array(y_true_online_list),
            y_pred_val    = np.array(y_pred_val_list),
            y_true_val    = np.array(y_true_val_list),
        )
        print(f"\n结果已保存: {fname}")

    # ── 9. 打印单次结果 ───────────────────────────────────────────
    best_ep = int(np.argmin(model_init.STATS[1]))
    print("\n" + "=" * 60)
    print(f"  最终结果 — {dataset.upper()}")
    print("=" * 60)
    print(f"  初始训练阶段（validate）:")
    print(f"    最佳 RMSE:    {model_init.STATS[1, best_ep]:.6f}  (epoch {best_ep})")
    print(f"    最终 RMSE:    {model_init.STATS[1, -1]:.6f}")
    print(f"    最终 NegLogP: {model_init.STATS[2, -1]:.6f}")
    print(f"    最终 Coverage:{model_init.STATS[3, -1]:.4f}")
    print(f"\n  Online / Validation SMSE:")
    print(f"    Online     SMSE: {smse_online:.6f}")
    print(f"    Validation SMSE: {smse_val:.6f}")
    print(f"    （< 1 优于均值基线，越小越好）")
    print(f"\n  收敛超参:")
    print(f"    ls  = {model_init.params['ls']}")
    print(f"    σ0² = {float(model_init.params['σ0'])**2:.6f}")
    print(f"    σn² = {float(model_init.params['σn'])**2:.6f}")

    print(f"\nX_tr_s 范围: {X_tr_s.min(axis=0)} ~ {X_tr_s.max(axis=0)}")
    print(f"y_tr_s 范围: {y_tr_s.min():.4f} ~ {y_tr_s.max():.4f}")

    if OUTPUT_CONFIG['plot']:
        plot_results(model_init.STATS, dataset)

    if OUTPUT_CONFIG['plot_y']:
            plot_y_results(
                np.array(y_pred_online_list), np.array(y_true_online_list),
                np.array(y_pred_val_list),    np.array(y_true_val_list),
                dataset, M,
        )
    
    return {
        'model':       model_init,
        'smse_online': smse_online,
        'smse_val':    smse_val,
        'nlp':         float(model_init.STATS[2, -1]),
        'coverage':    float(model_init.STATS[3, -1]),
        'learn_rate':          TRAIN_CONFIG['lr'],
    }


# ================================================================
# ===== 主函数：数据集 × M 双重循环 ==============================
# ================================================================

def main():
    results = {}

    for dataset in DATASET_LIST:
        for M in M_LIST:
            print(f"\n{'='*60}")
            print(f"  开始训练  数据集={dataset.upper()}  M={M}")
            print(f"{'='*60}")
            results[(dataset, M)] = run_single(M, dataset)

    col_w   = max(len(d) for d in DATASET_LIST) + 2
    total_w = col_w + 57
    sep     = '=' * total_w

    print(f"\n{sep}")
    print(f"{'  汇总结果':^{total_w}}")
    print(f"{sep}")
    print(
        f"  {'Dataset':<{col_w}}  {'M':>4}  "
        f"{'Online SMSE':>12}  {'Val SMSE':>10}  "
        f"{'NLP':>10}  {'Coverage':>9}"
    )
    print(f"  {'-' * (total_w - 2)}")
    for (dataset, M), res in results.items():
        print(
            f"  {dataset.upper():<{col_w}}  {M:>4}  "
            f"{res['smse_online']:>12.6f}  "
            f"{res['smse_val']:>10.6f}  "
            f"{res['nlp']:>10.6f}  "
            f"{res['coverage']:>9.4f}"
            f"{res['learn_rate']:>9.4f}"
        )
    print(f"{sep}\n")

    save_results_csv(results)
    return results


if __name__ == '__main__':
    results = main()