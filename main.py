import os, glob, random, copy, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

dataset_names = ['Bank']
protected_attributes = ['age']
CSV_DIR = './dataset'
RESULTS_DIR = './results'
TARGET_COL_DEFAULT = 'Probability'
TARGET_COL_OVERRIDES = {}

os.makedirs(RESULTS_DIR, exist_ok=True)
MASTER_LOG_PATH   = os.path.join(RESULTS_DIR, 'master_log.csv')
HISTORY_LOG_PATH  = os.path.join(RESULTS_DIR, 'round_history_log.csv')

NUM_CLIENTS      = 10
LOCAL_EPOCHS     = 5
BATCH_SIZE       = 32
LR               = 0.001
TEST_SIZE        = 0.20

DEFAULT_ALPHA    = 1.5
DEFAULT_BETA     = 3.0
DEFAULT_EMA      = 0.7
DEFAULT_MU       = 0.5
DEFAULT_EPS_FAIR = 0.1

WGAN_EPOCHS  = 300
WGAN_LR      = 1e-4
N_CRITIC     = 5
CLIP_VALUE   = 0.01
WGAN_HIDDEN  = 64
WGAN_LATENT  = 32

NUM_ROUNDS_MAIN   = 30
SEEDS_MAIN     = [42, 43, 44, 45, 46]

class WGANGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),   nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden * 2),   nn.LeakyReLU(0.2),
            nn.Linear(hidden * 2, output_dim), nn.Sigmoid()
        )
    def forward(self, z): return self.net(z)


class WGANCritic(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden * 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden * 2, hidden),    nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)


def train_wgan(real_data_np, latent_dim=32, hidden=64, epochs=300, lr=1e-4,
               n_critic=5, clip=0.01, batch_size=32, seed=42, device=None):
    device = device or DEVICE
    torch.manual_seed(seed); np.random.seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    n, feat_dim = real_data_np.shape
    if n < 5:
        return None
    loader = DataLoader(TensorDataset(torch.tensor(real_data_np, dtype=torch.float32)),
                         batch_size=min(batch_size, n), shuffle=True, drop_last=False)
    G = WGANGenerator(latent_dim, feat_dim, hidden).to(device)
    C = WGANCritic(feat_dim, hidden).to(device)
    opt_G = optim.RMSprop(G.parameters(), lr=lr)
    opt_C = optim.RMSprop(C.parameters(), lr=lr)
    G.train(); C.train()
    for _ in range(epochs):
        for (rb,) in loader:
            rb = rb.to(device)
            bs = rb.size(0)
            if bs < 2:
                continue
            for _ in range(n_critic):
                z = torch.randn(bs, latent_dim, device=device)
                loss_C = -(C(rb).mean() - C(G(z).detach()).mean())
                opt_C.zero_grad(); loss_C.backward(); opt_C.step()
                for p in C.parameters(): p.data.clamp_(-clip, clip)
            loss_G = -C(G(torch.randn(bs, latent_dim, device=device))).mean()
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
    G.eval()
    return G


def generate_synthetic(generator, n_samples, latent_dim, real_data_np,
                        col_names, target_val, protected_val, target_col, protected_col, device=None):
    device = device or DEVICE
    if n_samples <= 0:
        return pd.DataFrame(columns=col_names)
    if generator is None:
        idx = np.random.choice(len(real_data_np), n_samples, replace=True)
        synth = np.clip(real_data_np[idx] + np.random.normal(0, 0.01, real_data_np[idx].shape), 0, 1)
    else:
        with torch.no_grad():
            synth = generator(torch.randn(n_samples, latent_dim, device=device)).cpu().numpy()
    feat_cols = [c for c in col_names if c not in [target_col, protected_col]]
    df_s = pd.DataFrame(synth, columns=feat_cols)
    df_s[target_col] = target_val
    df_s[protected_col] = protected_val
    return df_s[col_names]


class GlobalMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 1),         nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)


def fairness_metrics(y_true, y_pred, protected):
    y_true = np.array(y_true); y_pred = np.array(y_pred); protected = np.array(protected)
    priv_idx = protected == 1
    unpriv_idx = protected == 0
    p_priv   = np.mean(y_pred[priv_idx] == 1)   if priv_idx.sum()   > 0 else 0.0
    p_unpriv = np.mean(y_pred[unpriv_idx] == 1) if unpriv_idx.sum() > 0 else 0.0
    SPD = p_unpriv - p_priv
    DI  = 1.0 - (p_unpriv / p_priv) if p_priv > 1e-9 else np.nan

    def rates(yt, yp):
        if len(yt) == 0: return 0.0, 0.0
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        tpr = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        fpr = fp/(fp+tn) if (fp+tn) > 0 else 0.0
        return tpr, fpr

    tpr_p, fpr_p = rates(y_true[priv_idx],   y_pred[priv_idx])
    tpr_u, fpr_u = rates(y_true[unpriv_idx], y_pred[unpriv_idx])
    EOD = tpr_u - tpr_p
    AOD = ((fpr_u - fpr_p) + (tpr_u - tpr_p)) / 2
    return {'DI': DI, 'SPD': SPD, 'EOD': EOD, 'AOD': AOD}


def full_eval(model, X_tensor, y_np, prot_np):
    model.eval()
    with torch.no_grad():
        preds = (model(X_tensor).squeeze() >= 0.5).float().cpu().numpy()
    acc  = accuracy_score(y_np, preds)
    bacc = balanced_accuracy_score(y_np, preds)
    fm   = fairness_metrics(y_np.astype(int), preds.astype(int), prot_np)
    return acc, bacc, fm


def is_pareto_nondominated(costs):
    N = costs.shape[0]
    dominated = np.zeros(N, dtype=bool)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                dominated[i] = True
                break
    return ~dominated


def run_pipeline(df, protected_attr, target_col,
                  alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, ema_lambda=DEFAULT_EMA,
                  mu=DEFAULT_MU, eps_fair=DEFAULT_EPS_FAIR,
                  num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS_MAIN,
                  local_epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                  seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed)
    EPS_NUM = 1e-9

    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=seed, shuffle=True)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    clients_raw = np.array_split(train_df, num_clients)

    X_test_np = test_df.drop(columns=[target_col]).values.astype(np.float32)
    y_test_np = test_df[target_col].values.astype(np.float32)
    prot_test = test_df[protected_attr].values
    X_test_tensor = torch.tensor(X_test_np).to(DEVICE)
    input_dim = X_test_np.shape[1]

    all_cols = clients_raw[0].columns.tolist()
    feat_cols_nn = [c for c in all_cols if c not in [target_col, protected_attr]]

    balanced_clients = []
    for cid, cdf in enumerate(clients_raw):
        subgroups = {(t, p): cdf[(cdf[target_col] == t) & (cdf[protected_attr] == p)]
                               .copy().reset_index(drop=True)
                     for t in [0, 1] for p in [0, 1]}
        max_size = max(len(v) for v in subgroups.values())
        augmented = []
        for (t, p), sub_df in subgroups.items():
            deficit = max_size - len(sub_df)
            if deficit > 0:
                real_np = sub_df[feat_cols_nn].values.astype(np.float32)
                gen = train_wgan(real_np, latent_dim=WGAN_LATENT, hidden=WGAN_HIDDEN,
                                  epochs=WGAN_EPOCHS, lr=WGAN_LR, n_critic=N_CRITIC,
                                  clip=CLIP_VALUE, seed=seed + cid*10 + t*2 + p, device=DEVICE)
                synth_df = generate_synthetic(gen, deficit, WGAN_LATENT, real_np, all_cols,
                                               t, p, target_col, protected_attr, device=DEVICE)
                sub_df = pd.concat([sub_df, synth_df], ignore_index=True)
            augmented.append(sub_df)
        bal_df = pd.concat(augmented, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
        bal_df[target_col] = bal_df[target_col].round().astype(int)
        bal_df[protected_attr] = bal_df[protected_attr].round().astype(int)
        balanced_clients.append(bal_df)

    global_model = GlobalMLP(input_dim).to(DEVICE)
    bce_loss = nn.BCELoss()
    ema_di = {i: 0.0 for i in range(num_clients)}
    candidate_pool = []
    round_history = []

    for rnd in range(1, num_rounds + 1):
        local_weights, round_acc, round_bacc = [], [], []
        round_di_signed, round_di_ema, round_spd, round_eod, round_aod, client_n = [], [], [], [], [], []

        for cid in range(num_clients):
            local_model = GlobalMLP(input_dim).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            local_model.train()
            opt = optim.Adam(local_model.parameters(), lr=lr)

            bal_df = balanced_clients[cid]
            X_bal = torch.tensor(bal_df.drop(columns=[target_col]).values.astype(np.float32))
            y_bal = torch.tensor(bal_df[target_col].values.astype(np.float32))
            loader = DataLoader(TensorDataset(X_bal, y_bal), batch_size=batch_size,
                                 shuffle=True, drop_last=False)

            for _ in range(local_epochs):
                for bX, by in loader:
                    if bX.size(0) < 2:
                        continue
                    bX, by = bX.to(DEVICE), by.to(DEVICE)
                    opt.zero_grad()
                    out = local_model(bX).squeeze()
                    if out.dim() == 0: out = out.unsqueeze(0)
                    loss_ce = bce_loss(out, by)
                    fair_penalty = mu * max(0.0, abs(ema_di[cid]) - eps_fair)
                    (loss_ce + fair_penalty).backward()
                    opt.step()

            l_acc, l_bacc, l_fm = full_eval(local_model, X_test_tensor, y_test_np, prot_test)
            l_di = l_fm['DI']
            if np.isnan(l_di): l_di = ema_di[cid]

            new_ema = ema_lambda * ema_di[cid] + (1 - ema_lambda) * l_di
            ema_di[cid] = new_ema

            round_acc.append(l_acc); round_bacc.append(l_bacc)
            round_di_signed.append(l_di); round_di_ema.append(new_ema)
            round_spd.append(l_fm['SPD']); round_eod.append(l_fm['EOD']); round_aod.append(l_fm['AOD'])
            client_n.append(len(bal_df))
            local_weights.append(copy.deepcopy(local_model.state_dict()))

        raw_scores = np.array([
            (round_acc[i] ** alpha) * np.exp(-beta * abs(round_di_ema[i]))
            for i in range(num_clients)
        ], dtype=np.float64)
        agg_weights = raw_scores / (raw_scores.sum() + EPS_NUM)

        best_cid = int(np.argmin([abs(d) for d in round_di_signed]))
        candidate_pool.append({
            'label': f'Client{best_cid+1}_R{rnd}', 'type': 'client', 'round': rnd,
            'acc': round_acc[best_cid], 'bacc': round_bacc[best_cid],
            'di_abs': abs(round_di_signed[best_cid]), 'di_signed': round_di_signed[best_cid],
            'spd': round_spd[best_cid], 'eod': round_eod[best_cid], 'aod': round_aod[best_cid],
            'state_dict': None,
        })

        global_dict = global_model.state_dict()
        for key in global_dict:
            global_dict[key] = sum(agg_weights[i] * local_weights[i][key].float()
                                    for i in range(num_clients))
        global_model.load_state_dict(global_dict)

        g_acc, g_bacc, g_fm = full_eval(global_model, X_test_tensor, y_test_np, prot_test)
        g_di = g_fm['DI'] if not np.isnan(g_fm['DI']) else 0.0
        candidate_pool.append({
            'label': f'Global_R{rnd}', 'type': 'global', 'round': rnd,
            'acc': g_acc, 'bacc': g_bacc,
            'di_abs': abs(g_di), 'di_signed': g_di,
            'spd': g_fm['SPD'], 'eod': g_fm['EOD'], 'aod': g_fm['AOD'],
            'state_dict': None,
        })

        round_history.append({
            'round': rnd,
            'global_acc': g_acc, 'global_bacc': g_bacc, 'global_di1': 1 - g_di,
            'global_spd': g_fm['SPD'], 'global_eod': g_fm['EOD'], 'global_aod': g_fm['AOD'],
            'mean_client_di1': float(np.mean([1 - d for d in round_di_signed])),
            'best_client_id':   best_cid + 1,
            'best_client_acc':  round_acc[best_cid],
            'best_client_bacc': round_bacc[best_cid],
            'best_client_di1':  1 - round_di_signed[best_cid],
            'best_client_spd':  round_spd[best_cid],
            'best_client_eod':  round_eod[best_cid],
            'best_client_aod':  round_aod[best_cid],
        })

    accs  = np.array([c['acc']    for c in candidate_pool])
    baccs = np.array([c['bacc']   for c in candidate_pool])
    dis   = np.array([c['di_abs'] for c in candidate_pool])
    cost_acc, cost_bacc = 1 - accs, 1 - baccs
    front_v1 = is_pareto_nondominated(np.column_stack([cost_acc, dis]))
    front_v2 = is_pareto_nondominated(np.column_stack([cost_bacc, dis]))
    front_v3 = is_pareto_nondominated(np.column_stack([cost_acc, cost_bacc, dis]))
    combined = front_v1 | front_v2 | front_v3
    composite = (accs ** alpha) * (baccs ** 0.5) * np.exp(-beta * dis)
    composite_masked = np.where(combined, composite, -np.inf)
    best_idx = int(np.argmax(composite_masked))
    best_candidate = candidate_pool[best_idx]

    result = {
        'selected_label':  best_candidate['label'],
        'selected_type':   best_candidate['type'],
        'selected_round':  best_candidate['round'],
        'final_acc':       best_candidate['acc'],
        'final_bacc':      best_candidate['bacc'],
        'final_di1':       1 - best_candidate['di_signed'],
        'final_spd':       best_candidate['spd'],
        'final_eod':       best_candidate['eod'],
        'final_aod':       best_candidate['aod'],
        'round_history':   round_history,
    }
    return result


def get_csv_path(dataset_name):
    return os.path.join(CSV_DIR, f'{dataset_name.lower()}_processed.csv')


def load_dataset(dataset_name, protected_attr):
    path = get_csv_path(dataset_name)
    df = pd.read_csv(path)
    target_col = TARGET_COL_OVERRIDES.get(dataset_name, TARGET_COL_DEFAULT)
    feature_cols = [c for c in df.columns if c not in [target_col, protected_attr]]
    numeric_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns
    scaler = MinMaxScaler()
    df = df.copy()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    df[target_col] = df[target_col].astype(int)
    df[protected_attr] = df[protected_attr].astype(int)
    return df, target_col


def _norm(v):
    return '' if v is None else str(v)


def load_completed_keys():
    if not os.path.exists(MASTER_LOG_PATH):
        return set()
    log = pd.read_csv(MASTER_LOG_PATH)
    keys = set(zip(
        log['dataset'], log['protected_attr'], log['experiment_type'], log['variant'],
        log['param_name'].fillna('').astype(str), log['param_value'].fillna('').astype(str),
        log['seed'].astype(str)
    ))
    return keys


def append_master_row(row: dict):
    df_row = pd.DataFrame([row])
    header = not os.path.exists(MASTER_LOG_PATH)
    df_row.to_csv(MASTER_LOG_PATH, mode='a', header=header, index=False)


def append_history_rows(dataset, protected_attr, experiment_type, variant,
                         param_name, param_value, seed, round_history):
    rows = []
    for rh in round_history:
        r = dict(rh)
        r.update({
            'dataset': dataset, 'protected_attr': protected_attr,
            'experiment_type': experiment_type, 'variant': variant,
            'param_name': _norm(param_name), 'param_value': _norm(param_value),
            'seed': seed,
        })
        rows.append(r)
    df_rows = pd.DataFrame(rows)
    header = not os.path.exists(HISTORY_LOG_PATH)
    df_rows.to_csv(HISTORY_LOG_PATH, mode='a', header=header, index=False)


if __name__ == '__main__':
    completed = load_completed_keys()

    def already_done(key):
        return key in completed

    def mark_done(key):
        completed.add(key)

    def make_key(dataset, protected_attr, experiment_type, variant, param_name, param_value, seed):
        return (dataset, protected_attr, experiment_type, variant,
                _norm(param_name), _norm(param_value), str(seed))

    for ds_name, prot_attr in zip(dataset_names, protected_attributes):
        ds_label = f'{ds_name}_{prot_attr}'

        try:
            df, target_col = load_dataset(ds_name, prot_attr)
        except FileNotFoundError:
            continue

        for seed in SEEDS_MAIN:
            key = make_key(ds_label, prot_attr, 'main', 'full', None, None, seed)
            if already_done(key):
                continue
            res = run_pipeline(df, prot_attr, target_col, num_rounds=NUM_ROUNDS_MAIN, seed=seed)
            append_master_row({
                'dataset': ds_label, 'protected_attr': prot_attr,
                'experiment_type': 'main', 'method': 'Proposed', 'variant': 'full',
                'param_name': '', 'param_value': '', 'seed': seed,
                'selected_round': res['selected_round'], 'selected_type': res['selected_type'],
                'acc': res['final_acc'], 'bacc': res['final_bacc'], 'di1': res['final_di1'],
                'spd': res['final_spd'], 'eod': res['final_eod'], 'aod': res['final_aod'],
            })
            append_history_rows(ds_label, prot_attr, 'main', 'full', None, None, seed, res['round_history'])
            mark_done(key)
