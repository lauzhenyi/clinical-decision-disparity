from __future__ import annotations

# Delete-a-group jackknife for the ICU no-chief-complaint-missing sensitivity analysis.

import concurrent.futures as cf
import hashlib
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from torch import nn
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
ICU_MAIN = ROOT / "main" / "icu_no_cc"
DATASET_NAMES = ["pressor", "vent"]

N_GROUPS = 20
LEFT_SCHEME = "structured"
RIGHT_SCHEME = "structured_plus_text"
SUBJECT_COL = "subject_id"
TEXT_SOURCE_COL = "chiefcomplaint"
KEYWORD_SELECTION_TOPK = 50

GROUP_VARS = ["gender", "race", "language"]
MODEL_METRIC_COLS = ["beta", "HR"]
WEIGHTED_OUTPUT_METRIC_COLS = [
    "beta_structured",
    "beta_structured_plus_text",
    "delta_beta",
    "HR_structured",
    "HR_structured_plus_text",
    "HR_ratio",
]

device = torch.device("cpu")
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)

G_ALL_POINT_STORE = None
G_ALL_REF = None
G_ALL_VEC = None
G_ALL_SVD = None
G_ALL_TRAIN_CFG = None
G_ALL_KEYWORD_META = None

def stable_seed(*parts):
    payload = "||".join(map(str, parts)).encode("utf-8")
    return int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16) % (2**31 - 1)

def safe_nanquantile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))

def safe_nanstat(x, fn, default=np.nan):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(default)
    return float(fn(x))

def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return np.nan
    return float(np.sum(w[mask] * x[mask]) / np.sum(w[mask]))

def make_train_val_split(n: int, seed: int, val_frac: float):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(np.floor(n * val_frac)))
    val_idx = np.sort(order[:n_val])
    train_idx = np.sort(order[n_val:])
    return train_idx, val_idx

def split_group_and_level(term: str):
    term = str(term)
    for g in GROUP_VARS:
        prefix = f"{g}_"
        if term.startswith(prefix):
            return g, term[len(prefix):]
    return None, term

def orient_gamma_demo_by_text(gamma: np.ndarray, demo_cols: list[str]):
    gamma = np.asarray(gamma, dtype=float)
    n_demo = int(len(demo_cols))
    if gamma.ndim != 2:
        raise ValueError(f"gamma must be 2D, got shape={gamma.shape}")
    if gamma.shape[0] == n_demo:
        return gamma
    if gamma.shape[1] == n_demo:
        return gamma.T
    raise ValueError(f"gamma shape={gamma.shape} is incompatible with n_demo={n_demo}")

class AdditiveCoxNetText(nn.Module):
    def __init__(self, n_base: int, n_text: int, n_demo: int, hidden=(64, 32), dropout=0.1):
        super().__init__()
        self.n_base = int(n_base)
        self.n_text = int(n_text)
        self.n_demo = int(n_demo)

        layers = []
        in_dim = self.n_base
        for h in hidden:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, 1))
        self.base_mlp = nn.Sequential(*layers)

        self.alpha = nn.Linear(self.n_demo, 1, bias=False)
        self.beta = nn.Linear(self.n_text, 1, bias=False)
        self.gamma = nn.Linear(self.n_demo, self.n_text, bias=False)

    def forward(self, x):
        xb = x[:, : self.n_base]
        z = x[:, self.n_base : self.n_base + self.n_text]
        d = x[:, self.n_base + self.n_text : self.n_base + self.n_text + self.n_demo]

        base_part = self.base_mlp(xb)
        demo_main = self.alpha(d)
        shared_text = self.beta(z)
        demo_text_int = (self.gamma(d) * z).sum(dim=1, keepdim=True)
        return base_part + demo_main + shared_text + demo_text_int

class StructuredCoxNet(nn.Module):
    def __init__(self, n_base: int, n_demo: int, hidden=(64, 32), dropout=0.1):
        super().__init__()
        self.n_base = int(n_base)
        self.n_demo = int(n_demo)

        layers = []
        in_dim = self.n_base
        for h in hidden:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, 1))
        self.base_mlp = nn.Sequential(*layers)
        self.alpha = nn.Linear(self.n_demo, 1, bias=False)

    def forward(self, x):
        xb = x[:, : self.n_base]
        d = x[:, self.n_base : self.n_base + self.n_demo]
        return self.base_mlp(xb) + self.alpha(d)

def _binned_weighted_cox_loss(log_hz, durations, events, weights):
    order = torch.argsort(durations, descending=True)
    t = durations[order]
    e = events[order]
    w = weights[order]
    lp = log_hz[order].reshape(-1)

    if torch.sum(e) <= 0:
        return torch.zeros((), device=lp.device)

    shift = torch.max(lp.detach())
    risk_term = w * torch.exp(lp - shift)
    risk_cumsum = torch.cumsum(risk_term, dim=0)

    _, counts = torch.unique_consecutive(t, return_counts=True)
    ends = torch.cumsum(counts, dim=0) - 1
    group_id = torch.repeat_interleave(torch.arange(len(counts), device=lp.device), counts)

    event_w = w * e
    group_event_w = torch.zeros(len(counts), device=lp.device, dtype=lp.dtype)
    group_event_lp = torch.zeros(len(counts), device=lp.device, dtype=lp.dtype)

    group_event_w.scatter_add_(0, group_id, event_w)
    group_event_lp.scatter_add_(0, group_id, event_w * lp)

    valid = group_event_w > 0
    denom = risk_cumsum[ends][valid]

    loss = -torch.sum(group_event_lp[valid])
    loss = loss + torch.sum(group_event_w[valid] * (torch.log(denom) + shift))

    total_event_weight = torch.sum(event_w)
    return loss / torch.clamp(total_event_weight, min=1e-8)

class WeightedCoxPHLoss(torch.nn.Module):
    def __init__(self, net=None, lambda_beta=0.0, lambda_gamma=0.0):
        super().__init__()
        self.net = net
        self.lambda_beta = float(lambda_beta)
        self.lambda_gamma = float(lambda_gamma)

    def forward(self, log_hz, durations, events, weights):
        loss = _binned_weighted_cox_loss(log_hz, durations, events, weights)

        penalty = torch.zeros((), device=log_hz.device)
        if self.net is not None and hasattr(self.net, "beta"):
            penalty = penalty + self.lambda_beta * torch.sum(self.net.beta.weight ** 2)
        if self.net is not None and hasattr(self.net, "gamma"):
            penalty = penalty + self.lambda_gamma * torch.sum(self.net.gamma.weight ** 2)

        return loss + penalty

class NeuralCoxModel:
    def __init__(self, net, model_kind: str):
        self.net = net.to(device)
        self.model_kind = model_kind
        self.baseline_event_times_ = None
        self.baseline_cumhaz_ = None
        self.train_info_ = {}
        self.tt_model = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["tt_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "tt_model" not in self.__dict__:
            self.tt_model = None
        if hasattr(self, "net") and self.net is not None:
            self.net = self.net.to(device)

    def predict(self, X):
        net_device = next(self.net.parameters()).device
        x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=net_device)
        self.net.eval()
        with torch.no_grad():
            out = self.net(x).reshape(-1).detach().cpu().numpy()
        return out

def fit_weighted_neural_cox(
    X: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    sample_weight: np.ndarray,
    train_weight: np.ndarray,
    val_weight: np.ndarray,
    model_kind: str,
    n_demo: int,
    n_text: int,
    hidden=(64, 32),
    dropout=0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 256,
    lambda_beta: float = 1e-4,
    lambda_gamma: float = 5e-4,
):
    X_np = X.to_numpy(np.float32)
    d_np = np.asarray(durations, dtype=np.float32)
    e_np = np.asarray(events, dtype=np.float32)
    w_train = np.asarray(train_weight, dtype=np.float32)
    w_val = np.asarray(val_weight, dtype=np.float32)

    train_mask = w_train > 0
    val_mask = w_val > 0

    w_scale = float(np.mean(w_train[train_mask])) if train_mask.any() else 1.0
    if not np.isfinite(w_scale) or w_scale <= 0:
        w_scale = 1.0

    w_train = w_train / w_scale
    w_val = w_val / w_scale

    if model_kind == "structured":
        n_base = X_np.shape[1] - int(n_demo)
        net = StructuredCoxNet(n_base=n_base, n_demo=n_demo, hidden=hidden, dropout=dropout)
    else:
        n_base = X_np.shape[1] - int(n_demo) - int(n_text)
        net = AdditiveCoxNetText(
            n_base=n_base,
            n_text=n_text,
            n_demo=n_demo,
            hidden=hidden,
            dropout=dropout,
        )

    loss = WeightedCoxPHLoss(net=net, lambda_beta=lambda_beta, lambda_gamma=lambda_gamma)
    tt_model = tt.Model(
        net,
        loss=loss,
        optimizer=tt.optim.Adam(lr=lr, weight_decay=weight_decay),
        device=device,
    )

    x_train = X_np[train_mask]
    y_train = (d_np[train_mask], e_np[train_mask], w_train[train_mask].astype(np.float32))

    x_val = X_np[val_mask]
    y_val = (d_np[val_mask], e_np[val_mask], w_val[val_mask].astype(np.float32))

    log = tt_model.fit(
        x_train,
        y_train,
        batch_size=max(1, len(x_train)),
        epochs=int(epochs),
        callbacks=[],
        verbose=False,
        shuffle=False,
        val_data=(x_val, y_val),
        val_batch_size=max(1, len(x_val)),
    )

    log_df = log.to_pandas().reset_index(drop=True)
    if "val_loss" in log_df.columns:
        val_loss = pd.to_numeric(log_df["val_loss"], errors="coerce").to_numpy(dtype=float)
        val_loss_finite = np.isfinite(val_loss)
        if np.any(val_loss_finite):
            min_idx = int(np.nanargmin(val_loss))
            min_val_loss = float(val_loss[min_idx])
        else:
            min_idx = -1
            min_val_loss = np.nan
        final_val_loss = float(val_loss[-1]) if val_loss.size else np.nan
    else:
        min_idx = -1
        min_val_loss = np.nan
        final_val_loss = np.nan

    model = NeuralCoxModel(net=net, model_kind=model_kind)
    model.tt_model = tt_model
    model.train_info_ = {
        "min_val_loss": min_val_loss,
        "min_val_epoch": int(min_idx) if min_idx >= 0 else -1,
        "final_val_loss": final_val_loss,
        "n_epoch_run": int(len(log_df)),
        "weight_mean_train": float(w_scale),
    }
    return model

def disparity_table_from_model_counterfactual(
    model: NeuralCoxModel,
    X: pd.DataFrame,
    ref_map: dict,
    sample_weight: np.ndarray | None = None,
    families=("gender_", "race_", "language_"),
):
    cols = X.columns.to_list()
    x_ref_template = X.copy()
    rows = []

    if sample_weight is None:
        sample_weight = np.ones(len(X), dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if len(sample_weight) != len(X):
            raise ValueError(
                f"sample_weight length mismatch: weight={len(sample_weight)} rows={len(X)}"
            )

    for prefix in families:
        fam_cols = [c for c in cols if c.startswith(prefix)]
        if len(fam_cols) == 0:
            continue

        group_var = prefix[:-1]
        ref_level = ref_map.get(group_var)
        X_ref = x_ref_template.copy()
        X_ref.loc[:, fam_cols] = 0.0

        lp_ref = model.predict(X_ref)
        rows.append(
            {
                "group_var": group_var,
                "level": ref_level,
                "ref_level": ref_level,
                "term": f"{group_var}_{ref_level}",
                "beta": 0.0,
                "HR": 1.0,
            }
        )

        for fam_col in fam_cols:
            X_cf = X_ref.copy()
            X_cf.loc[:, fam_col] = 1.0
            lp_cf = model.predict(X_cf)
            beta = weighted_mean(lp_cf - lp_ref, sample_weight)
            level = fam_col.split("_", 1)[1]

            rows.append(
                {
                    "group_var": group_var,
                    "level": level,
                    "ref_level": ref_level,
                    "term": fam_col,
                    "beta": beta,
                    "HR": float(np.exp(beta)),
                }
            )

    out = pd.DataFrame(rows)
    return out.sort_values(["group_var", "level"]).reset_index(drop=True)

def compare_disparity(tbl_left: pd.DataFrame, tbl_right: pd.DataFrame, left_scheme: str, right_scheme: str):
    join_cols = ["group_var", "level", "ref_level", "term"]

    left = tbl_left[join_cols + MODEL_METRIC_COLS].rename(
        columns={c: f"{c}_{left_scheme}" for c in MODEL_METRIC_COLS}
    )
    right = tbl_right[join_cols + MODEL_METRIC_COLS].rename(
        columns={c: f"{c}_{right_scheme}" for c in MODEL_METRIC_COLS}
    )

    out = left.merge(right, on=join_cols, how="inner")
    out["left_scheme"] = left_scheme
    out["right_scheme"] = right_scheme
    out["delta_beta"] = out[f"beta_{right_scheme}"] - out[f"beta_{left_scheme}"]
    out["HR_ratio"] = out[f"HR_{right_scheme}"] / out[f"HR_{left_scheme}"]
    out["abs_delta_beta"] = out["delta_beta"].abs()
    return out

def build_weighted_point_compare_results(
    artifact_store: dict,
    ref_map: dict,
    left_scheme: str = LEFT_SCHEME,
    right_scheme: str = RIGHT_SCHEME,
):
    rows = []

    for outcome, by_family in artifact_store.items():
        for family_var, by_scheme in by_family.items():
            if left_scheme not in by_scheme or right_scheme not in by_scheme:
                continue

            scheme_tables = {}
            tau = np.nan

            for scheme_name in [left_scheme, right_scheme]:
                artifact = by_scheme[scheme_name]
                model = artifact["model"]
                X_model = artifact["X_model"].reset_index(drop=True).copy()
                sample_weight = np.asarray(artifact["sample_weight"], dtype=float)

                tbl = disparity_table_from_model_counterfactual(
                    model=model,
                    X=X_model,
                    ref_map=ref_map,
                    families=(f"{family_var}_",),
                    sample_weight=sample_weight,
                )
                tbl["scheme"] = scheme_name
                tbl["family_var"] = family_var
                tbl["n_risk"] = int(len(X_model))
                tbl["n_event"] = int(np.sum(np.asarray(artifact["event"], dtype=float)))
                tbl["weight_sum"] = float(np.sum(sample_weight))
                scheme_tables[scheme_name] = tbl

                if "table" in artifact and artifact["table"] is not None:
                    table = artifact["table"]
                    if isinstance(table, pd.DataFrame) and "tau" in table.columns and len(table) > 0:
                        tau = pd.to_numeric(table["tau"], errors="coerce").iloc[0]

            cmp = compare_disparity(
                tbl_left=scheme_tables[left_scheme].copy(),
                tbl_right=scheme_tables[right_scheme].copy(),
                left_scheme=left_scheme,
                right_scheme=right_scheme,
            )
            cmp["outcome"] = outcome
            cmp["family_var"] = family_var
            cmp["tau"] = tau
            rows.append(cmp)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    base_cols = ["outcome", "tau", "family_var", "group_var", "level", "ref_level", "term"]
    ordered_cols = [col for col in base_cols if col in out.columns] + [
        col for col in out.columns if col not in base_cols
    ]
    return out.loc[:, ordered_cols].reset_index(drop=True)

def recover_text_standardization_spec(artifact: dict):
    text_cols = list(artifact["text_cols"])
    if len(text_cols) == 0:
        return None

    if "z_spec" in artifact and artifact["z_spec"] is not None:
        z_spec = artifact["z_spec"]
        return {
            "mu": np.asarray(z_spec["mu"], dtype=float),
            "sd": np.asarray(z_spec["sd"], dtype=float),
        }

    fit_df = artifact["fit_df"].reset_index(drop=True)
    X_model = artifact["X_model"].reset_index(drop=True)

    raw = fit_df[text_cols].to_numpy(dtype=float)
    std = X_model[text_cols].to_numpy(dtype=float)

    raw_mean = raw.mean(axis=0)
    std_mean = std.mean(axis=0)
    raw_sd = raw.std(axis=0, ddof=0)
    std_sd = std.std(axis=0, ddof=0)

    z_sd = np.ones(len(text_cols), dtype=float)
    mask = np.isfinite(std_sd) & (std_sd > 0)
    z_sd[mask] = raw_sd[mask] / std_sd[mask]
    z_mu = raw_mean - z_sd * std_mean
    return {"mu": z_mu.astype(float), "sd": z_sd.astype(float)}

def make_demo_group_frame(demo_cols: list[str], ref_map: dict):
    rows = []
    for demo_col in demo_cols:
        group_var, level = split_group_and_level(demo_col)
        rows.append(
            {
                "group_var": group_var,
                "level": level,
                "ref_level": ref_map.get(group_var),
                "term": demo_col,
            }
        )
    return pd.DataFrame(rows)

def compute_reference_token_statistics(artifact: dict, vec):
    fit_df = artifact["fit_df"].reset_index(drop=True)
    if TEXT_SOURCE_COL not in fit_df.columns:
        raise KeyError(f"{TEXT_SOURCE_COL} is required in fit_df for keyword contribution decomposition")

    text = fit_df[TEXT_SOURCE_COL].astype("string").fillna("").astype(str).tolist()
    tfidf_ref = vec.transform(text).astype(np.float32).tocsr()
    n_row = int(tfidf_ref.shape[0])

    token_ref_mean = np.asarray(tfidf_ref.mean(axis=0)).ravel().astype(float)
    token_ref_document_frequency = (tfidf_ref.getnnz(axis=0).astype(float) / float(n_row)).astype(float)
    return tfidf_ref, token_ref_mean, token_ref_document_frequency


def mean_tfidf_over_mask(tfidf_ref, mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return np.full(tfidf_ref.shape[1], np.nan, dtype=np.float32), 0

    mean = np.asarray(tfidf_ref[mask].mean(axis=0)).ravel().astype(np.float32)
    return mean, int(mask.sum())


def make_family_column_lookup(demo_cols: list[str]):
    family_col_idx = {}
    demo_col_index = {}
    for j, col in enumerate(demo_cols):
        group_var, _ = split_group_and_level(col)
        family_col_idx.setdefault(group_var, []).append(j)
        demo_col_index[col] = j

    return {
        group_var: np.asarray(idx, dtype=int)
        for group_var, idx in family_col_idx.items()
    }, demo_col_index


def compute_group_token_statistics(
    tfidf_ref,
    demo_indicator: np.ndarray,
    group_frame: pd.DataFrame,
    family_col_idx: dict,
    demo_col_index: dict,
    row_mask: np.ndarray | None = None,
):
    demo_indicator = np.asarray(demo_indicator)
    if row_mask is None:
        row_mask = np.ones(demo_indicator.shape[0], dtype=bool)
    else:
        row_mask = np.asarray(row_mask, dtype=bool)

    n_group = len(group_frame)
    n_token = int(tfidf_ref.shape[1])

    group_token_mean = np.full((n_group, n_token), np.nan, dtype=np.float32)
    ref_token_mean = np.full((n_group, n_token), np.nan, dtype=np.float32)
    group_row_count = np.zeros(n_group, dtype=np.int32)
    ref_row_count = np.zeros(n_group, dtype=np.int32)

    for j, row in enumerate(group_frame.itertuples(index=False)):
        fam_idx = family_col_idx[row.group_var]
        demo_idx = demo_col_index[row.term]

        group_mask = (demo_indicator[:, demo_idx] > 0.5) & row_mask
        ref_mask = np.all(demo_indicator[:, fam_idx] < 0.5, axis=1) & row_mask

        group_mean_j, group_n_j = mean_tfidf_over_mask(tfidf_ref, group_mask)
        ref_mean_j, ref_n_j = mean_tfidf_over_mask(tfidf_ref, ref_mask)

        group_token_mean[j] = group_mean_j
        ref_token_mean[j] = ref_mean_j
        group_row_count[j] = int(group_n_j)
        ref_row_count[j] = int(ref_n_j)

    token_mean_diff = group_token_mean - ref_token_mean
    return {
        "group_token_mean": group_token_mean,
        "ref_token_mean": ref_token_mean,
        "token_mean_diff": token_mean_diff,
        "group_row_count": group_row_count,
        "ref_row_count": ref_row_count,
    }

def align_gamma_demo_to_reference(gamma, demo_cols_current: list[str], demo_cols_ref: list[str]):
    gamma_demo = orient_gamma_demo_by_text(gamma, demo_cols_current)
    current_map = {col: j for j, col in enumerate(demo_cols_current)}
    out = np.zeros((len(demo_cols_ref), gamma_demo.shape[1]), dtype=float)
    for j_ref, col in enumerate(demo_cols_ref):
        if col in current_map:
            out[j_ref] = gamma_demo[current_map[col]]
    return out


def save_df_pair(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    df.to_csv(path.with_suffix(".csv"), index=False)

def build_keyword_metadata_store(artifact_store: dict, vec, svd, ref_map: dict):
    vocab = np.asarray(vec.get_feature_names_out())
    outcome_meta = {}
    token_rows = []

    for outcome, by_family in artifact_store.items():
        for family_var, by_scheme in by_family.items():
            if RIGHT_SCHEME not in by_scheme:
                continue

            artifact = by_scheme[RIGHT_SCHEME]
            text_cols = list(artifact["text_cols"])
            if len(text_cols) == 0:
                continue

            z_spec = recover_text_standardization_spec(artifact)
            projection_matrix = np.asarray(svd.components_, dtype=float).T / z_spec["sd"].reshape(1, -1)
            tfidf_ref, token_ref_mean, token_ref_document_frequency = compute_reference_token_statistics(artifact, vec)
            group_frame = make_demo_group_frame(list(artifact["demo_cols"]), ref_map)
            group_frame = group_frame.loc[group_frame["group_var"] == family_var].reset_index(drop=True)

            demo_cols = list(artifact["demo_cols"])
            demo_indicator = artifact["X_model"].reset_index(drop=True)[demo_cols].to_numpy(dtype=np.float32)
            family_col_idx, demo_col_index = make_family_column_lookup(demo_cols)
            point_group_stats = compute_group_token_statistics(
                tfidf_ref=tfidf_ref,
                demo_indicator=demo_indicator,
                group_frame=group_frame,
                family_col_idx=family_col_idx,
                demo_col_index=demo_col_index,
                row_mask=None,
            )

            store_key = (outcome, family_var)
            outcome_meta[store_key] = {
                "outcome": outcome,
                "family_var": family_var,
                "vocab": vocab,
                "token_id": np.arange(len(vocab), dtype=np.int32),
                "demo_cols": demo_cols,
                "group_frame": group_frame,
                "projection_matrix": projection_matrix.astype(np.float32),
                "token_ref_mean": token_ref_mean.astype(np.float32),
                "token_ref_document_frequency": token_ref_document_frequency.astype(np.float32),
                "z_mu": np.asarray(z_spec["mu"], dtype=np.float32),
                "z_sd": np.asarray(z_spec["sd"], dtype=np.float32),
                "tfidf_ref": tfidf_ref,
                "demo_indicator": demo_indicator.astype(np.float32),
                "family_col_idx": family_col_idx,
                "demo_col_index": demo_col_index,
                "group_token_mean": point_group_stats["group_token_mean"].astype(np.float32),
                "ref_token_mean": point_group_stats["ref_token_mean"].astype(np.float32),
                "token_mean_diff": point_group_stats["token_mean_diff"].astype(np.float32),
                "group_row_count": point_group_stats["group_row_count"].astype(np.int32),
                "ref_row_count": point_group_stats["ref_row_count"].astype(np.int32),
            }

            token_rows.append(
                pd.DataFrame(
                    {
                        "outcome": outcome,
                        "family_var": family_var,
                        "token_id": np.arange(len(vocab), dtype=np.int32),
                        "token": vocab,
                        "token_ref_mean": token_ref_mean.astype(np.float32),
                        "token_ref_document_frequency": token_ref_document_frequency.astype(np.float32),
                    }
                )
            )

    token_metadata = pd.concat(token_rows, ignore_index=True) if token_rows else pd.DataFrame()
    return outcome_meta, token_metadata

def build_keyword_point_outputs(artifact_store: dict, keyword_meta_store: dict):
    point_rows = []
    point_store = {}

    for store_key, meta in keyword_meta_store.items():
        outcome = meta["outcome"]
        family_var = meta["family_var"]
        artifact = artifact_store[outcome][family_var][RIGHT_SCHEME]
        beta_z = np.asarray(artifact["beta_z"], dtype=float)
        gamma_demo = align_gamma_demo_to_reference(
            artifact["gamma"],
            demo_cols_current=list(artifact["demo_cols"]),
            demo_cols_ref=list(meta["demo_cols"]),
        )
        M = np.asarray(meta["projection_matrix"], dtype=float)
        token_ref_mean = np.asarray(meta["token_ref_mean"], dtype=float)
        group_token_mean = np.asarray(meta["group_token_mean"], dtype=float)
        ref_token_mean = np.asarray(meta["ref_token_mean"], dtype=float)
        token_mean_diff = np.asarray(meta["token_mean_diff"], dtype=float)

        shared_score_vec = beta_z @ M.T
        shared_score = np.repeat(shared_score_vec.reshape(1, -1), len(meta["group_frame"]), axis=0)
        interaction_score = gamma_demo @ M.T
        group_frame = meta["group_frame"].reset_index(drop=True)
        keep_idx = [list(meta["demo_cols"]).index(term) for term in group_frame["term"].tolist()]
        interaction_score = interaction_score[keep_idx]
        total_score = interaction_score + shared_score
        contribution_score = interaction_score * token_ref_mean.reshape(1, -1)
        mix_contribution = token_mean_diff * shared_score
        interaction_contribution = group_token_mean * interaction_score
        total_contribution = mix_contribution + interaction_contribution

        point_store[store_key] = {
            "shared_score": shared_score.astype(np.float32),
            "interaction_score": interaction_score.astype(np.float32),
            "total_score": total_score.astype(np.float32),
            "contribution_score": contribution_score.astype(np.float32),
            "group_token_mean": group_token_mean.astype(np.float32),
            "ref_token_mean": ref_token_mean.astype(np.float32),
            "token_mean_diff": token_mean_diff.astype(np.float32),
            "mix_contribution": mix_contribution.astype(np.float32),
            "interaction_contribution": interaction_contribution.astype(np.float32),
            "total_contribution": total_contribution.astype(np.float32),
        }

        token_id = meta["token_id"]
        vocab = meta["vocab"]
        for j, row in group_frame.iterrows():
            point_rows.append(
                pd.DataFrame(
                    {
                        "outcome": outcome,
                        "family_var": family_var,
                        "group_var": row["group_var"],
                        "level": row["level"],
                        "ref_level": row["ref_level"],
                        "term": row["term"],
                        "group_row_count": int(meta["group_row_count"][j]),
                        "ref_row_count": int(meta["ref_row_count"][j]),
                        "token_id": token_id,
                        "token": vocab,
                        "shared_score": shared_score[j].astype(np.float32),
                        "interaction_score": interaction_score[j].astype(np.float32),
                        "total_score": total_score[j].astype(np.float32),
                        "contribution_score": contribution_score[j].astype(np.float32),
                        "group_token_mean": group_token_mean[j].astype(np.float32),
                        "ref_token_mean": ref_token_mean[j].astype(np.float32),
                        "token_mean_diff": token_mean_diff[j].astype(np.float32),
                        "mix_contribution": mix_contribution[j].astype(np.float32),
                        "interaction_contribution": interaction_contribution[j].astype(np.float32),
                        "total_contribution": total_contribution[j].astype(np.float32),
                    }
                )
            )

    point_df = pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame()
    return point_store, point_df

def jackknife_summary_table(
    replicate_df: pd.DataFrame,
    point_df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
    rep_col: str = "delete_group",
):
    rows = []

    for keys, g in replicate_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: key for col, key in zip(group_cols, keys)}
        row["jackknife_g"] = int(g[rep_col].nunique())

        m = row["jackknife_g"]
        for col in metric_cols:
            x = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            row[f"{col}_jackknife_mean"] = float(np.mean(x)) if x.size else np.nan

            if x.size >= 2:
                se2 = ((m - 1.0) / m) * np.sum((x - np.mean(x)) ** 2)
                row[f"{col}_jackknife_se"] = float(np.sqrt(se2))
            else:
                row[f"{col}_jackknife_se"] = np.nan

            row[f"{col}_jackknife_rep_min"] = float(np.min(x)) if x.size else np.nan
            row[f"{col}_jackknife_rep_max"] = float(np.max(x)) if x.size else np.nan

        rows.append(row)

    summary = pd.DataFrame(rows)
    out = point_df.merge(summary, on=group_cols, how="left")

    for col in metric_cols:
        out[f"{col}_jackknife_bias"] = (out["jackknife_g"] - 1.0) * (out[f"{col}_jackknife_mean"] - out[col])
        out[f"{col}_jackknife_bias_corrected"] = out[col] - out[f"{col}_jackknife_bias"]
        out[f"{col}_jackknife_ci_low"] = out[col] - 1.96 * out[f"{col}_jackknife_se"]
        out[f"{col}_jackknife_ci_high"] = out[col] + 1.96 * out[f"{col}_jackknife_se"]

    return out

def build_keyword_replicate_store(keyword_payloads: list[dict], keyword_meta_store: dict):
    grouped = {}
    for payload in keyword_payloads:
        if payload is None:
            continue
        grouped.setdefault(payload["store_key"], []).append(payload)

    metric_cols = [
        "shared_score",
        "interaction_score",
        "total_score",
        "contribution_score",
        "group_token_mean",
        "ref_token_mean",
        "token_mean_diff",
        "mix_contribution",
        "interaction_contribution",
        "total_contribution",
    ]

    out = {}
    for store_key, rows in grouped.items():
        rows = sorted(rows, key=lambda x: x["delete_group"])
        meta = keyword_meta_store[store_key]
        out[store_key] = {
            "outcome": meta["outcome"],
            "family_var": meta["family_var"],
            "delete_group": np.asarray([row["delete_group"] for row in rows], dtype=np.int32),
            "group_frame": meta["group_frame"].copy(),
            "token_id": meta["token_id"].copy(),
            "vocab": meta["vocab"].copy(),
        }

        for metric_col in metric_cols:
            out[store_key][metric_col] = np.stack([row[metric_col] for row in rows], axis=0).astype(np.float32)

    return out

def compute_topk_selection_frequency(arr: np.ndarray, top_k: int):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.full(arr.shape[1], np.nan, dtype=np.float32)

    k = min(int(top_k), int(arr.shape[1]))
    counts = np.zeros(arr.shape[1], dtype=np.int32)
    valid_rep_n = 0
    for b in range(arr.shape[0]):
        row = np.asarray(arr[b], dtype=float)
        finite = np.isfinite(row)
        if finite.sum() == 0:
            continue

        valid_rep_n += 1
        score = np.full_like(row, -np.inf, dtype=float)
        score[finite] = np.abs(row[finite])
        idx = np.argpartition(-score, k - 1)[:k]
        counts[idx] += 1

    if valid_rep_n == 0:
        return np.full(arr.shape[1], np.nan, dtype=np.float32)

    return (counts / float(valid_rep_n)).astype(np.float32)

def make_keyword_metric_summary(keyword_replicate_store: dict, keyword_point_store: dict, metric_col: str, top_k: int):
    rows = []

    for store_key, rep in keyword_replicate_store.items():
        outcome = rep["outcome"]
        family_var = rep["family_var"]
        group_frame = rep["group_frame"].reset_index(drop=True)
        token_id = np.asarray(rep["token_id"], dtype=np.int32)
        vocab = np.asarray(rep["vocab"])
        x = np.asarray(rep[metric_col], dtype=float)
        point = np.asarray(keyword_point_store[store_key][metric_col], dtype=float)

        finite = np.isfinite(x)
        jackknife_g = finite.sum(axis=0).astype(np.int32)

        sum_x = np.where(finite, x, 0.0).sum(axis=0)
        mean = np.divide(
            sum_x,
            np.maximum(jackknife_g, 1),
            out=np.full_like(sum_x, np.nan, dtype=float),
            where=jackknife_g > 0,
        )

        centered = np.where(finite, x - mean[None, :, :], 0.0)
        se = np.sqrt(
            np.divide(
                np.maximum(jackknife_g - 1, 0),
                np.maximum(jackknife_g, 1),
                out=np.zeros_like(mean, dtype=float),
                where=jackknife_g > 1,
            )
            * np.sum(centered ** 2, axis=0)
        )
        se = np.where(jackknife_g > 1, se, np.nan)

        rep_min = np.where(finite.any(axis=0), np.nanmin(x, axis=0), np.nan)
        rep_max = np.where(finite.any(axis=0), np.nanmax(x, axis=0), np.nan)

        for j, row in group_frame.iterrows():
            select_freq = compute_topk_selection_frequency(x[:, j, :], top_k=top_k)
            point_j = point[j]
            mean_j = mean[j]
            se_j = se[j]
            rep_min_j = rep_min[j]
            rep_max_j = rep_max[j]
            jackknife_g_j = jackknife_g[j].astype(np.int32)
            bias_j = np.where(
                jackknife_g_j > 0,
                (jackknife_g_j - 1.0) * (mean_j - point_j),
                np.nan,
            )
            bias_corrected_j = point_j - bias_j

            rows.append(
                pd.DataFrame(
                    {
                        "outcome": outcome,
                        "family_var": family_var,
                        "group_var": row["group_var"],
                        "level": row["level"],
                        "ref_level": row["ref_level"],
                        "term": row["term"],
                        "token_id": token_id,
                        "token": vocab,
                        metric_col: point_j.astype(np.float32),
                        "jackknife_g": jackknife_g_j,
                        f"{metric_col}_jackknife_mean": mean_j.astype(np.float32),
                        f"{metric_col}_jackknife_se": se_j.astype(np.float32),
                        f"{metric_col}_jackknife_rep_min": rep_min_j.astype(np.float32),
                        f"{metric_col}_jackknife_rep_max": rep_max_j.astype(np.float32),
                        f"{metric_col}_jackknife_bias": bias_j.astype(np.float32),
                        f"{metric_col}_jackknife_bias_corrected": bias_corrected_j.astype(np.float32),
                        f"{metric_col}_jackknife_ci_low": (point_j - 1.96 * se_j).astype(np.float32),
                        f"{metric_col}_jackknife_ci_high": (point_j + 1.96 * se_j).astype(np.float32),
                        f"{metric_col}_selection_frequency_topk": select_freq,
                    }
                )
            )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def get_bundle_dir(dataset_name: str) -> Path:
    return ICU_MAIN / dataset_name / "dagjackknife_bundle"

def get_output_dir(dataset_name: str) -> Path:
    out_dir = ICU_MAIN / dataset_name / "delete_a_group_jackknife"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def load_bundle(dataset_name: str):
    bundle_dir = get_bundle_dir(dataset_name)

    compare_results = pd.read_parquet(bundle_dir / "compare_results.parquet")

    with open(bundle_dir / "main_artifacts.pkl", "rb") as f:
        main_artifacts = pickle.load(f)

    vec = joblib.load(bundle_dir / "vec.joblib")
    svd = joblib.load(bundle_dir / "svd.joblib")

    with open(bundle_dir / "REF.pkl", "rb") as f:
        ref_map = pickle.load(f)

    with open(bundle_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_cfg = {
        "DATASET": str(cfg.get("DATASET", dataset_name)),
        "VAL_FRAC": float(cfg["VAL_FRAC"]),
        "LR": float(cfg["LR"]),
        "WEIGHT_DECAY": float(cfg["WEIGHT_DECAY"]),
        "EPOCHS": int(cfg["EPOCHS"]),
        "PATIENCE": int(cfg.get("PATIENCE", 0)),
        "LAMBDA_BETA": float(cfg["LAMBDA_BETA"]),
        "LAMBDA_GAMMA": float(cfg["LAMBDA_GAMMA"]),
        "LEFT_SCHEME": str(cfg["LEFT_SCHEME"]),
        "RIGHT_SCHEME": str(cfg["RIGHT_SCHEME"]),
        "SUBJECT_COL": str(cfg["SUBJECT_COL"]),
    }
    return compare_results, main_artifacts, vec, svd, ref_map, train_cfg

def make_fixed_cluster_split_with_event_guard(
    cluster_event_n,
    val_frac: float,
    seed: int,
    max_tries: int = 128,
):
    cluster_event_n = np.asarray(cluster_event_n, dtype=float)
    n_cluster = int(len(cluster_event_n))

    for k in range(int(max_tries)):
        tr_idx, val_idx = make_train_val_split(
            n_cluster,
            seed=int(seed) + int(k),
            val_frac=val_frac,
        )

        tr_idx = np.asarray(tr_idx, dtype=int)
        val_idx = np.asarray(val_idx, dtype=int)

        if len(tr_idx) == 0 or len(val_idx) == 0:
            continue

        train_event = float(cluster_event_n[tr_idx].sum())
        val_event = float(cluster_event_n[val_idx].sum())

        if train_event > 0.0 and val_event > 0.0:
            cluster_is_train = np.zeros(n_cluster, dtype=bool)
            cluster_is_val = np.zeros(n_cluster, dtype=bool)
            cluster_is_train[tr_idx] = True
            cluster_is_val[val_idx] = True
            return cluster_is_train, cluster_is_val

    raise RuntimeError("Could not construct a fixed cluster split with positive event weight in both splits.")

def make_subject_cluster_refit_store_fixed(
    artifact_store: dict,
    compare_df: pd.DataFrame,
    train_cfg: dict,
    left_scheme: str = LEFT_SCHEME,
    right_scheme: str = RIGHT_SCHEME,
    subject_col: str = SUBJECT_COL,
):
    point_store = {}

    for outcome, by_family in artifact_store.items():
        for family_var, by_scheme in by_family.items():
            if left_scheme not in by_scheme or right_scheme not in by_scheme:
                continue

            left_artifact = by_scheme[left_scheme]
            right_artifact = by_scheme[right_scheme]

            left_fit_df = left_artifact["fit_df"].reset_index(drop=True).copy()
            right_fit_df = right_artifact["fit_df"].reset_index(drop=True).copy()

            if len(left_fit_df) != len(right_fit_df):
                raise ValueError(
                    f"fit_df length mismatch for outcome={outcome}, family={family_var}: "
                    f"left={len(left_fit_df)} right={len(right_fit_df)}"
                )

            left_subject = left_fit_df[subject_col].astype("string").reset_index(drop=True)
            right_subject = right_fit_df[subject_col].astype("string").reset_index(drop=True)
            if not left_subject.equals(right_subject):
                raise ValueError(f"subject alignment mismatch between schemes for outcome={outcome}, family={family_var}")

            compare_sub = compare_df.loc[
                (compare_df["outcome"] == outcome)
                & (compare_df["group_var"] == family_var)
            ].reset_index(drop=True).copy()
            if len(compare_sub) == 0:
                continue

            subject_ids = right_subject.to_numpy()
            subject_codes, subject_levels = pd.factorize(subject_ids, sort=False)

            cluster_row_n = np.bincount(subject_codes, minlength=len(subject_levels)).astype(float)
            cluster_event_n = np.bincount(
                subject_codes,
                weights=np.asarray(right_artifact["event"], dtype=float),
                minlength=len(subject_levels),
            ).astype(float)

            cluster_is_train, cluster_is_val = make_fixed_cluster_split_with_event_guard(
                cluster_event_n=cluster_event_n,
                val_frac=train_cfg["VAL_FRAC"],
                seed=stable_seed("icu_refit_fixed_cluster_split", train_cfg["DATASET"], outcome, family_var),
                max_tries=256,
            )
            row_is_train = cluster_is_train[subject_codes]
            row_is_val = cluster_is_val[subject_codes]

            scheme_payload = {}
            for scheme_name in [left_scheme, right_scheme]:
                artifact = by_scheme[scheme_name]

                X_ref = artifact["X_model"].reset_index(drop=True).copy()
                duration = np.asarray(artifact["duration"], dtype=float)
                event = np.asarray(artifact["event"], dtype=np.int8)
                sample_weight = np.asarray(artifact["sample_weight"], dtype=float)
                demo_cols = list(artifact["demo_cols"])
                text_cols = list(artifact["text_cols"])

                if len(X_ref) != len(subject_ids):
                    raise ValueError(
                        f"X_model length mismatch for outcome={outcome}, family={family_var}, scheme={scheme_name}: "
                        f"X={len(X_ref)} subjects={len(subject_ids)}"
                    )

                if not (len(duration) == len(event) == len(sample_weight) == len(subject_ids)):
                    raise ValueError(
                        f"vector length mismatch for outcome={outcome}, family={family_var}, scheme={scheme_name}"
                    )

                scheme_payload[scheme_name] = {
                    "X_ref": X_ref,
                    "duration": duration,
                    "event": event,
                    "sample_weight": sample_weight,
                    "demo_cols": demo_cols,
                    "text_cols": text_cols,
                    "model_kind": "structured" if len(text_cols) == 0 else "text",
                    "n_demo": int(len(demo_cols)),
                    "n_text": int(len(text_cols)),
                }

            store_key = (outcome, family_var)
            point_store[store_key] = {
                "dataset": str(train_cfg["DATASET"]),
                "outcome": outcome,
                "family_var": family_var,
                "compare_point": compare_sub.copy(),
                "tau": pd.to_numeric(compare_sub["tau"], errors="coerce").iloc[0] if "tau" in compare_sub.columns else np.nan,
                "subject_ids": subject_ids,
                "subject_codes": subject_codes.astype(int),
                "subject_levels": pd.Index(subject_levels).astype("string").to_numpy(),
                "cluster_row_n": cluster_row_n,
                "cluster_event_n": cluster_event_n,
                "cluster_is_train": cluster_is_train,
                "cluster_is_val": cluster_is_val,
                "row_is_train": row_is_train,
                "row_is_val": row_is_val,
                "n_rows": int(cluster_row_n.sum()),
                "n_event": int(cluster_event_n.sum()),
                "n_clusters": int(len(subject_levels)),
                "schemes": scheme_payload,
            }

    return point_store

def make_delete_a_group_partitions(point_store: dict, n_groups: int, dataset_name: str):
    partitions = {}
    group_rows = []

    for store_key, payload in point_store.items():
        outcome = payload["outcome"]
        family_var = payload["family_var"]
        n_clusters = int(payload["n_clusters"])
        order = np.random.default_rng(
            stable_seed("icu_delete_a_group_partition", dataset_name, outcome, family_var, n_groups)
        ).permutation(n_clusters)
        split_idx = np.array_split(order, n_groups)
        partitions[store_key] = [np.asarray(x, dtype=int) for x in split_idx]

        for delete_group, cluster_idx in enumerate(split_idx):
            deleted_subject_ids = payload["subject_levels"][cluster_idx]
            group_rows.append(
                {
                    "dataset": dataset_name,
                    "outcome": outcome,
                    "family_var": family_var,
                    "delete_group": int(delete_group),
                    "cluster_unit": SUBJECT_COL,
                    "n_total_clusters": n_clusters,
                    "n_deleted_clusters": int(len(cluster_idx)),
                    "n_deleted_rows": int(payload["cluster_row_n"][cluster_idx].sum()),
                    "n_deleted_events": int(payload["cluster_event_n"][cluster_idx].sum()),
                    "deleted_subject_id_preview": "|".join(map(str, deleted_subject_ids[:10])),
                }
            )

    group_table = pd.DataFrame(group_rows)
    return partitions, group_table

def fit_delete_group_refit_scheme_fixed(
    outcome: str,
    family_var: str,
    scheme_name: str,
    payload: dict,
    cluster_keep: np.ndarray,
    ref_map: dict,
    train_cfg: dict,
):
    subject_codes = np.asarray(payload["subject_codes"], dtype=int)
    row_keep = cluster_keep[subject_codes].astype(float)

    scheme_payload = payload["schemes"][scheme_name]
    base_weight = np.asarray(scheme_payload["sample_weight"], dtype=float)

    full_weight = base_weight * row_keep
    train_weight = full_weight * np.asarray(payload["row_is_train"], dtype=float)
    val_weight = full_weight * np.asarray(payload["row_is_val"], dtype=float)

    train_mask = train_weight > 0
    val_mask = val_weight > 0

    event = np.asarray(scheme_payload["event"], dtype=np.int8)
    n_train_event = int(event[train_mask].sum())
    n_val_event = int(event[val_mask].sum())

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        return None, {
            "diagnostic_type": "delete_group_skipped",
            "dataset": payload["dataset"],
            "outcome": outcome,
            "family_var": family_var,
            "scheme": scheme_name,
            "skip_reason": "empty_train_or_val_after_group_deletion",
        }, None

    if n_train_event == 0 or n_val_event == 0:
        return None, {
            "diagnostic_type": "delete_group_skipped",
            "dataset": payload["dataset"],
            "outcome": outcome,
            "family_var": family_var,
            "scheme": scheme_name,
            "skip_reason": "zero_event_in_train_or_val_after_group_deletion",
        }, None

    init_seed = stable_seed("icu_refit_fixed_init", payload["dataset"], outcome, family_var, scheme_name)
    np.random.seed(int(init_seed))
    torch.manual_seed(int(init_seed))

    model = fit_weighted_neural_cox(
        X=scheme_payload["X_ref"],
        durations=scheme_payload["duration"],
        events=scheme_payload["event"],
        sample_weight=full_weight,
        train_weight=train_weight,
        val_weight=val_weight,
        model_kind=scheme_payload["model_kind"],
        n_demo=scheme_payload["n_demo"],
        n_text=scheme_payload["n_text"],
        hidden=(64, 32),
        dropout=0.1,
        lr=train_cfg["LR"],
        weight_decay=train_cfg["WEIGHT_DECAY"],
        epochs=train_cfg["EPOCHS"],
        lambda_beta=train_cfg["LAMBDA_BETA"],
        lambda_gamma=train_cfg["LAMBDA_GAMMA"],
    )

    tbl = disparity_table_from_model_counterfactual(
        model=model,
        X=scheme_payload["X_ref"],
        ref_map=ref_map,
        families=(f"{family_var}_",),
        sample_weight=full_weight,
    )
    tbl["scheme"] = scheme_name
    tbl["family_var"] = family_var
    tbl["n_risk"] = int(round(np.sum(row_keep)))
    tbl["n_event"] = int(round(np.sum(cluster_keep * payload["cluster_event_n"])))
    tbl["weight_sum"] = float(np.sum(full_weight))

    diagnostics = {
        "diagnostic_type": "delete_group_model_fit",
        "dataset": payload["dataset"],
        "outcome": outcome,
        "family_var": family_var,
        "scheme": scheme_name,
        "n_rows_reference": int(len(scheme_payload["X_ref"])),
        "n_rows_positive_weight": int(np.sum(full_weight > 0)),
        "n_demo_cols": int(scheme_payload["n_demo"]),
        "n_text_cols": int(scheme_payload["n_text"]),
        "n_train_rows_positive_weight": int(train_mask.sum()),
        "n_val_rows_positive_weight": int(val_mask.sum()),
        "n_train_event_positive_weight": n_train_event,
        "n_val_event_positive_weight": n_val_event,
        "weight_min": safe_nanstat(full_weight, np.min),
        "weight_p01": safe_nanquantile(full_weight, 0.01),
        "weight_p50": safe_nanquantile(full_weight, 0.50),
        "weight_p99": safe_nanquantile(full_weight, 0.99),
        "weight_max": safe_nanstat(full_weight, np.max),
        "effective_sample_size": float((np.sum(full_weight) ** 2) / np.sum(full_weight ** 2)),
        "min_val_loss": float(model.train_info_["min_val_loss"]),
        "min_val_epoch": int(model.train_info_["min_val_epoch"]),
        "final_val_loss": float(model.train_info_["final_val_loss"]),
        "n_epoch_run": int(model.train_info_["n_epoch_run"]),
    }

    artifact = {
        "model": model,
        "X_model": scheme_payload["X_ref"],
        "fit_df": pd.DataFrame({"subject_id": payload["subject_ids"]}),
        "duration": scheme_payload["duration"],
        "event": scheme_payload["event"],
        "sample_weight": full_weight,
        "demo_cols": list(scheme_payload["demo_cols"]),
        "text_cols": list(scheme_payload["text_cols"]),
        "family_var": family_var,
        "alpha_demo": model.net.alpha.weight.detach().cpu().numpy().reshape(-1),
        "beta_z": None,
        "gamma": None,
        "table": tbl.copy(),
    }

    if scheme_payload["model_kind"] == "text":
        artifact["beta_z"] = model.net.beta.weight.detach().cpu().numpy().reshape(-1)
        artifact["gamma"] = model.net.gamma.weight.detach().cpu().numpy()

    return tbl, diagnostics, artifact


def init_worker(all_point_store, all_ref, all_vec, all_svd, all_train_cfg, all_keyword_meta):
    global G_ALL_POINT_STORE, G_ALL_REF, G_ALL_VEC, G_ALL_SVD, G_ALL_TRAIN_CFG, G_ALL_KEYWORD_META
    G_ALL_POINT_STORE = all_point_store
    G_ALL_REF = all_ref
    G_ALL_VEC = all_vec
    G_ALL_SVD = all_svd
    G_ALL_TRAIN_CFG = all_train_cfg
    G_ALL_KEYWORD_META = all_keyword_meta

    torch.set_num_threads(1)


def run_one_delete_group_task(task):
    dataset = task["dataset"]
    store_key = task["store_key"]
    outcome, family_var = store_key
    delete_group = task["delete_group"]
    deleted_cluster_idx = np.asarray(task["deleted_cluster_idx"], dtype=int)

    payload = G_ALL_POINT_STORE[dataset][store_key]
    ref_map = G_ALL_REF[dataset]
    train_cfg = G_ALL_TRAIN_CFG[dataset]

    n_clusters = int(payload["n_clusters"])
    cluster_keep = np.ones(n_clusters, dtype=float)
    cluster_keep[deleted_cluster_idx] = 0.0
    subject_codes = np.asarray(payload["subject_codes"], dtype=int)
    row_keep = cluster_keep[subject_codes].astype(float)

    kept_rows = float(cluster_keep @ payload["cluster_row_n"])
    kept_events = float(cluster_keep @ payload["cluster_event_n"])
    kept_clusters = int(cluster_keep.sum())

    scheme_tables = []
    diagnostics_rows = []
    keyword_artifact = None

    for scheme_name in [LEFT_SCHEME, RIGHT_SCHEME]:
        tbl, diag, scheme_keyword_artifact = fit_delete_group_refit_scheme_fixed(
            outcome=outcome,
            family_var=family_var,
            scheme_name=scheme_name,
            payload=payload,
            cluster_keep=cluster_keep,
            ref_map=ref_map,
            train_cfg=train_cfg,
        )

        diag["delete_group"] = int(delete_group)
        diag["resampling_method"] = "delete_a_group_jackknife_fixed_ipsw"
        diagnostics_rows.append(diag)

        if tbl is None:
            return {
                "dataset": dataset,
                "compare": None,
                "counts": pd.DataFrame(
                    [
                        {
                            "dataset": dataset,
                            "outcome": outcome,
                            "family_var": family_var,
                            "tau": payload["tau"],
                            "delete_group": int(delete_group),
                            "n_risk": int(round(kept_rows)),
                            "n_event": int(round(kept_events)),
                            "n_kept_clusters": kept_clusters,
                            "n_deleted_clusters": int(len(deleted_cluster_idx)),
                            "status": "skipped",
                        }
                    ]
                ),
                "diagnostics": pd.DataFrame(diagnostics_rows),
                "keyword_payload": None,
            }

        tbl["dataset"] = dataset
        tbl["family_var"] = family_var
        tbl["delete_group"] = int(delete_group)
        scheme_tables.append(tbl)

        if scheme_name == RIGHT_SCHEME:
            keyword_artifact = scheme_keyword_artifact

    outcome_long = pd.concat(scheme_tables, ignore_index=True)
    cmp = compare_disparity(
        tbl_left=outcome_long.loc[outcome_long["scheme"] == LEFT_SCHEME].copy(),
        tbl_right=outcome_long.loc[outcome_long["scheme"] == RIGHT_SCHEME].copy(),
        left_scheme=LEFT_SCHEME,
        right_scheme=RIGHT_SCHEME,
    )
    cmp["dataset"] = dataset
    cmp["outcome"] = outcome
    cmp["family_var"] = family_var
    cmp["tau"] = payload["tau"]
    cmp["delete_group"] = int(delete_group)

    diagnostics_rows.append(
        {
            "diagnostic_type": "delete_group_draw",
            "dataset": dataset,
            "outcome": outcome,
            "family_var": family_var,
            "delete_group": int(delete_group),
            "resampling_method": "delete_a_group_jackknife_fixed_ipsw",
            "n_clusters_full": int(payload["n_clusters"]),
            "n_rows_full": int(payload["n_rows"]),
            "n_event_full": int(payload["n_event"]),
            "n_kept_clusters": kept_clusters,
            "n_deleted_clusters": int(len(deleted_cluster_idx)),
            "n_kept_rows": int(round(kept_rows)),
            "n_kept_events": int(round(kept_events)),
            "model_refit": True,
            "ipsw_refit": False,
            "text_refit": False,
            "feature_matrix_fixed": True,
            "reference_population_fixed": True,
        }
    )

    keyword_payload = None
    if keyword_artifact is not None and store_key in G_ALL_KEYWORD_META[dataset]:
        meta = G_ALL_KEYWORD_META[dataset][store_key]
        gamma_demo = align_gamma_demo_to_reference(
            keyword_artifact["gamma"],
            demo_cols_current=list(keyword_artifact["demo_cols"]),
            demo_cols_ref=list(meta["demo_cols"]),
        )
        M = np.asarray(meta["projection_matrix"], dtype=float)
        token_ref_mean = np.asarray(meta["token_ref_mean"], dtype=float)

        shared_score_vec = np.asarray(keyword_artifact["beta_z"], dtype=float) @ M.T
        shared_score = np.repeat(shared_score_vec.reshape(1, -1), len(meta["group_frame"]), axis=0)
        interaction_score_full = gamma_demo @ M.T
        keep_idx = [list(meta["demo_cols"]).index(term) for term in meta["group_frame"]["term"].tolist()]
        interaction_score = interaction_score_full[keep_idx]
        total_score = interaction_score + shared_score
        contribution_score = interaction_score * token_ref_mean.reshape(1, -1)

        keyword_group_stats = compute_group_token_statistics(
            tfidf_ref=meta["tfidf_ref"],
            demo_indicator=meta["demo_indicator"],
            group_frame=meta["group_frame"],
            family_col_idx=meta["family_col_idx"],
            demo_col_index=meta["demo_col_index"],
            row_mask=row_keep > 0,
        )
        group_token_mean = np.asarray(keyword_group_stats["group_token_mean"], dtype=float)
        ref_token_mean = np.asarray(keyword_group_stats["ref_token_mean"], dtype=float)
        token_mean_diff = np.asarray(keyword_group_stats["token_mean_diff"], dtype=float)

        mix_contribution = token_mean_diff * shared_score
        interaction_contribution = group_token_mean * interaction_score
        total_contribution = mix_contribution + interaction_contribution

        keyword_payload = {
            "store_key": store_key,
            "outcome": outcome,
            "family_var": family_var,
            "delete_group": int(delete_group),
            "shared_score": shared_score.astype(np.float32),
            "interaction_score": interaction_score.astype(np.float32),
            "total_score": total_score.astype(np.float32),
            "contribution_score": contribution_score.astype(np.float32),
            "group_token_mean": group_token_mean.astype(np.float32),
            "ref_token_mean": ref_token_mean.astype(np.float32),
            "token_mean_diff": token_mean_diff.astype(np.float32),
            "mix_contribution": mix_contribution.astype(np.float32),
            "interaction_contribution": interaction_contribution.astype(np.float32),
            "total_contribution": total_contribution.astype(np.float32),
        }

    counts_df = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "outcome": outcome,
                "family_var": family_var,
                "tau": payload["tau"],
                "delete_group": int(delete_group),
                "n_risk": int(round(kept_rows)),
                "n_event": int(round(kept_events)),
                "n_kept_clusters": kept_clusters,
                "n_deleted_clusters": int(len(deleted_cluster_idx)),
                "status": "ok",
            }
        ]
    )

    return {
        "dataset": dataset,
        "compare": cmp,
        "counts": counts_df,
        "diagnostics": pd.DataFrame(diagnostics_rows),
        "keyword_payload": keyword_payload,
    }

def main():
    all_point_store = {}
    all_ref = {}
    all_vec = {}
    all_svd = {}
    all_train_cfg = {}
    all_keyword_meta = {}
    all_keyword_point_store = {}
    all_keyword_point_df = {}
    all_keyword_token_metadata = {}
    all_compare_results = {}
    all_group_tables = {}
    tasks = []

    for dataset_name in DATASET_NAMES:
        bundle_dir = get_bundle_dir(dataset_name)
        if not bundle_dir.exists():
            raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

        compare_results, main_artifacts, vec, svd, ref_map, train_cfg = load_bundle(dataset_name)
        weighted_point_compare = build_weighted_point_compare_results(
            artifact_store=main_artifacts,
            ref_map=ref_map,
            left_scheme=LEFT_SCHEME,
            right_scheme=RIGHT_SCHEME,
        )

        keyword_meta_store, keyword_token_metadata = build_keyword_metadata_store(
            artifact_store=main_artifacts,
            vec=vec,
            svd=svd,
            ref_map=ref_map,
        )
        keyword_point_store, keyword_point_df = build_keyword_point_outputs(main_artifacts, keyword_meta_store)

        if len(keyword_token_metadata) > 0:
            keyword_token_metadata.insert(0, "dataset", dataset_name)
        if len(keyword_point_df) > 0:
            keyword_point_df.insert(0, "dataset", dataset_name)

        point_store = make_subject_cluster_refit_store_fixed(
            artifact_store=main_artifacts,
            compare_df=weighted_point_compare,
            train_cfg=train_cfg,
            left_scheme=LEFT_SCHEME,
            right_scheme=RIGHT_SCHEME,
            subject_col=SUBJECT_COL,
        )

        partitions, group_table = make_delete_a_group_partitions(
            point_store=point_store,
            n_groups=N_GROUPS,
            dataset_name=dataset_name,
        )

        all_point_store[dataset_name] = point_store
        all_ref[dataset_name] = ref_map
        all_vec[dataset_name] = vec
        all_svd[dataset_name] = svd
        all_train_cfg[dataset_name] = train_cfg
        all_keyword_meta[dataset_name] = keyword_meta_store
        all_keyword_point_store[dataset_name] = keyword_point_store
        all_keyword_point_df[dataset_name] = keyword_point_df
        all_keyword_token_metadata[dataset_name] = keyword_token_metadata
        all_compare_results[dataset_name] = weighted_point_compare
        all_group_tables[dataset_name] = group_table

        for store_key, split_idx in partitions.items():
            for delete_group, deleted_cluster_idx in enumerate(split_idx):
                tasks.append(
                    {
                        "dataset": dataset_name,
                        "store_key": store_key,
                        "delete_group": int(delete_group),
                        "deleted_cluster_idx": np.asarray(deleted_cluster_idx, dtype=int),
                    }
                )

    total_cores = os.cpu_count() or 1
    n_jobs = max(1, total_cores - 2)

    compare_parts = {name: [] for name in DATASET_NAMES}
    counts_parts = {name: [] for name in DATASET_NAMES}
    diagnostics_parts = {name: [] for name in DATASET_NAMES}
    keyword_payloads = {name: [] for name in DATASET_NAMES}

    with cf.ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=init_worker,
        initargs=(all_point_store, all_ref, all_vec, all_svd, all_train_cfg, all_keyword_meta),
    ) as ex:
        futures = [ex.submit(run_one_delete_group_task, task) for task in tasks]

        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Delete-a-group jackknife"):
            res = fut.result()
            dataset_name = res["dataset"]
            if res["compare"] is not None:
                compare_parts[dataset_name].append(res["compare"])
            counts_parts[dataset_name].append(res["counts"])
            diagnostics_parts[dataset_name].append(res["diagnostics"])
            if res["keyword_payload"] is not None:
                keyword_payloads[dataset_name].append(res["keyword_payload"])

    keyword_metric_defs = {
        "shared_score": "jackknife_keyword_shared_summary.parquet",
        "interaction_score": "jackknife_keyword_interaction_summary.parquet",
        "total_score": "jackknife_keyword_total_summary.parquet",
        "contribution_score": "jackknife_keyword_contribution_summary.parquet",
        "group_token_mean": "jackknife_keyword_group_token_mean_summary.parquet",
        "ref_token_mean": "jackknife_keyword_ref_token_mean_summary.parquet",
        "token_mean_diff": "jackknife_keyword_token_mean_diff_summary.parquet",
        "mix_contribution": "jackknife_keyword_mix_contribution_summary.parquet",
        "interaction_contribution": "jackknife_keyword_interaction_contribution_summary.parquet",
        "total_contribution": "jackknife_keyword_total_contribution_summary.parquet",
    }

    for dataset_name in DATASET_NAMES:
        out_dir = get_output_dir(dataset_name)
        compare_results = all_compare_results[dataset_name]
        group_table = all_group_tables[dataset_name]
        keyword_meta_store = all_keyword_meta[dataset_name]
        keyword_point_store = all_keyword_point_store[dataset_name]
        keyword_point_df = all_keyword_point_df[dataset_name]
        keyword_token_metadata = all_keyword_token_metadata[dataset_name]

        jackknife_compare = (
            pd.concat(compare_parts[dataset_name], ignore_index=True)
            if compare_parts[dataset_name]
            else pd.DataFrame()
        )
        jackknife_counts = (
            pd.concat(counts_parts[dataset_name], ignore_index=True)
            if counts_parts[dataset_name]
            else pd.DataFrame()
        )
        jackknife_diagnostics = (
            pd.concat(diagnostics_parts[dataset_name], ignore_index=True)
            if diagnostics_parts[dataset_name]
            else pd.DataFrame()
        )

        jackknife_point_compare = compare_results.copy()

        jackknife_summary = jackknife_summary_table(
            replicate_df=jackknife_compare,
            point_df=jackknife_point_compare,
            group_cols=["outcome", "family_var", "group_var", "level", "ref_level", "term"],
            metric_cols=WEIGHTED_OUTPUT_METRIC_COLS,
            rep_col="delete_group",
        )

        keyword_replicate_store = build_keyword_replicate_store(keyword_payloads[dataset_name], keyword_meta_store)
        keyword_summary_tables = {}
        for metric_col in keyword_metric_defs:
            keyword_summary_tables[metric_col] = make_keyword_metric_summary(
                keyword_replicate_store=keyword_replicate_store,
                keyword_point_store=keyword_point_store,
                metric_col=metric_col,
                top_k=KEYWORD_SELECTION_TOPK,
            )
            if len(keyword_summary_tables[metric_col]) > 0:
                keyword_summary_tables[metric_col].insert(0, "dataset", dataset_name)

        save_df_pair(jackknife_point_compare, out_dir / "jackknife_point_compare.parquet")
        save_df_pair(jackknife_compare, out_dir / "jackknife_compare.parquet")
        save_df_pair(jackknife_summary, out_dir / "jackknife_summary.parquet")
        save_df_pair(jackknife_counts, out_dir / "jackknife_counts.parquet")
        save_df_pair(jackknife_diagnostics, out_dir / "jackknife_diagnostics.parquet")
        save_df_pair(group_table, out_dir / "jackknife_group_table.parquet")
        save_df_pair(keyword_token_metadata, out_dir / "jackknife_keyword_token_metadata.parquet")
        save_df_pair(keyword_point_df, out_dir / "jackknife_keyword_point.parquet")

        for metric_col, filename in keyword_metric_defs.items():
            save_df_pair(keyword_summary_tables[metric_col], out_dir / filename)

        with open(out_dir / "jackknife_keyword_replicate_store.pkl", "wb") as f:
            pickle.dump(keyword_replicate_store, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Saved delete-a-group jackknife outputs to:", out_dir)
        print("point_compare:", out_dir / "jackknife_point_compare.parquet")
        print("compare:", out_dir / "jackknife_compare.parquet")
        print("summary:", out_dir / "jackknife_summary.parquet")
        print("counts:", out_dir / "jackknife_counts.parquet")
        print("diagnostics:", out_dir / "jackknife_diagnostics.parquet")
        print("groups:", out_dir / "jackknife_group_table.parquet")
        print("keyword_token_metadata:", out_dir / "jackknife_keyword_token_metadata.parquet")
        print("keyword_point:", out_dir / "jackknife_keyword_point.parquet")
        for metric_col, filename in keyword_metric_defs.items():
            print(f"{metric_col}:", out_dir / filename)
        print("keyword_replicate_store:", out_dir / "jackknife_keyword_replicate_store.pkl")

if __name__ == "__main__":
    main()
