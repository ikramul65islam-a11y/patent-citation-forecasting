"""Resumable 20-split evaluation of the six tuned patent-citation models.

Run from a directory containing the full-panel inputs and selected-settings JSONs:

    python fullpanel_final_20split.py --data-dir . --output-dir final20 --dry-run
    python fullpanel_final_20split.py --data-dir . --output-dir final20

The dry run verifies files, alignment, configurations, and a tiny GRU forward
pass. It does not fit a model or score test records. Completed predictions are
verified and reused on subsequent runs. Seeds 20--39 are robustness splits:
their test sets overlap the seed-40 training set used for hyperparameter choice.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SIZES = {1: 268192, 2: 248874, 3: 225128, 4: 201400, 5: 178765}
MODEL_NAMES = (
    "xgb_citations", "xgb_text", "xgb_centroids",
    "gru_citations", "gru_text", "gru_centroids",
)
FEATURE_NAMES = (
    "cit_0", "logcit_0", "cit_1", "logcit_1", "cit_2", "logcit_2",
    "cit_3", "logcit_3", "cit_4", "logcit_4", "total", "log_total",
    "mean", "max", "n_nonzero", "slope", "recent_2", "early_2",
    "accel", "recent_share", "com",
)


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_csv(frame, path):
    temp = path.with_suffix(".partial.csv")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def build_features(counts):
    x = pd.DataFrame(np.asarray(counts, dtype=np.float64), columns=range(5))
    f = pd.DataFrame(index=x.index)
    for age in range(5):
        f[f"cit_{age}"] = x[age]
        f[f"logcit_{age}"] = np.log1p(x[age])
    total = x.sum(axis=1)
    f["total"] = total
    f["log_total"] = np.log1p(total)
    f["mean"] = x.mean(axis=1)
    f["max"] = x.max(axis=1)
    f["n_nonzero"] = (x > 0).sum(axis=1)
    ages = np.arange(5, dtype=float)
    values = x.to_numpy(dtype=float)
    f["slope"] = ((values - values.mean(axis=1, keepdims=True))
                  * (ages - ages.mean())).sum(axis=1) / ((ages - ages.mean()) ** 2).sum()
    f["recent_2"] = x[[3, 4]].sum(axis=1)
    f["early_2"] = x[[0, 1]].sum(axis=1)
    f["accel"] = f["recent_2"] - f["early_2"]
    safe = np.where(total.to_numpy() > 0, total.to_numpy(), 1)
    f["recent_share"] = np.where(total.to_numpy() > 0,
                                  f["recent_2"].to_numpy() / safe, 0)
    f["com"] = np.where(total.to_numpy() > 0,
                        (values * ages).sum(axis=1) / safe, 0)
    assert tuple(f.columns) == FEATURE_NAMES
    result = f.to_numpy(dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite citation features")
    return result


def build_gru_inputs(counts):
    counts = np.asarray(counts, dtype=np.float32)
    total = counts.sum(axis=1)
    safe = np.where(total > 0, total, 1)
    ages = np.arange(5, dtype=np.float32)
    sequence = np.log1p(counts).astype(np.float32)
    scalars = np.column_stack((
        np.log1p(total),
        np.where(total > 0, (counts * ages).sum(axis=1) / safe, 0),
        np.where(total > 0, counts[:, 3:5].sum(axis=1) / safe, 0),
        (counts > 0).sum(axis=1),
    )).astype(np.float32)
    return sequence, scalars


def load_panel(data_dir, h1_ids, expected_hashes):
    """Keep only input/target ages 0--9 while validating the raw panel."""
    index = pd.Index(h1_ids)
    n = len(h1_ids)
    values = np.full((n, 10), -1, dtype=np.int32)
    min_year = np.full(n, 9999, dtype=np.int16)
    max_year = np.zeros(n, dtype=np.int16)
    rows_scanned = 0
    for chunk in pd.read_csv(
        data_dir / "citation_panel_han.csv",
        usecols=["record_id", "pub_year", "year", "age", "citations"],
        dtype={"record_id": np.int64, "pub_year": np.int16,
               "year": np.int16, "age": np.int16, "citations": np.int32},
        chunksize=50000,
    ):
        rows_scanned += len(chunk)
        if (chunk.age < 0).any() or (chunk.citations < 0).any():
            raise ValueError("Negative panel ages or counts")
        if not (chunk.year == chunk.pub_year + chunk.age).all():
            raise ValueError("Inconsistent panel calendar years")
        pos = index.get_indexer(chunk.record_id.to_numpy())
        age = chunk.age.to_numpy()
        keep = (pos >= 0) & (age <= 9)
        pos, age = pos[keep], age[keep]
        if not len(pos):
            continue
        keys = pos.astype(np.int64) * 10 + age
        if len(np.unique(keys)) != len(keys) or (values[pos, age] >= 0).any():
            raise ValueError("Duplicate patent-age rows")
        values[pos, age] = chunk.citations.to_numpy()[keep]
        years = chunk.pub_year.to_numpy()[keep]
        np.minimum.at(min_year, pos, years)
        np.maximum.at(max_year, pos, years)
    if (min_year != max_year).any() or (values[:, :6] < 0).any():
        raise ValueError("Inconsistent publication year or missing ages 0--5")
    if (min_year < 1999).any() or (min_year > 2020).any():
        raise ValueError("Unexpected eligible H=1 publication years")
    data = {}
    for horizon in range(1, 6):
        eligible = min_year <= 2025 - (4 + horizon)
        ids = h1_ids[eligible]
        window = values[eligible, :5 + horizon]
        if len(ids) != SIZES[horizon] or (window < 0).any():
            raise ValueError(f"H={horizon}: missing or miscounted histories")
        if ids_sha(ids) != expected_hashes[horizon]:
            raise ValueError(f"H={horizon}: IDs/order differ from tuning")
        data[horizon] = dict(
            ids=ids, h1_rows=np.flatnonzero(eligible),
            X=window[:, :5].astype(np.float32),
            y=window[:, 5:5+horizon].sum(axis=1, dtype=np.int64).astype(np.float32),
            yr=min_year[eligible].copy(),
        )
    print(f"Panel validated: {rows_scanned:,} rows; H=1--5 ID hashes match", flush=True)
    return data


def p10_pooled(y, p):
    k = int(len(y) * .10)
    if not k:
        raise ValueError("Too few patents for pooled P@10")
    return float((y[np.argsort(-p)[:k]] >= np.quantile(y, .90)).mean())


def p10_cohort(y, p, years):
    hits, selected = 0, 0
    for year in np.unique(years):
        mask = years == year
        if mask.sum() < 20:
            continue
        yc, pc = y[mask], p[mask]
        k = max(1, int(len(yc) * .10))
        hits += int((yc[np.argsort(-pc)[:k]] >= np.quantile(yc, .90)).sum())
        selected += k
    if not selected:
        raise ValueError("No cohort with at least 20 patents")
    return float(hits / selected)


def load_settings(data_dir, expected_hashes):
    files = {
        "xgb_citations": "fullpanel_xgb_citations_selected_settings_H1_to_H5.json",
        "xgb_text": "fullpanel_xgb_text_selected_settings.json",
        "xgb_centroids": "fullpanel_xgb_centroids_original_selected_settings_H1_to_H5.json",
        "gru_citations": "fullpanel_gru_citations_selected_H1_to_H5.json",
        "gru_text": "fullpanel_gru_text_selected_H1_to_H5.json",
        "gru_centroids": "fullpanel_gru_centroids_original_selected_H1_to_H5.json",
    }
    raw = {name: read_json(data_dir / file) for name, file in files.items()}
    candidates = read_json(data_dir / "fullpanel_gru_candidates.json")
    if isinstance(candidates, dict):
        raise ValueError("Expected the GRU candidates JSON to contain a list")
    if len(candidates) != 16:
        raise ValueError("Expected 16 GRU candidates")
    candidate_hash = sha(candidates)
    for model, key_name in (("gru_citations", "candidate_hash"),
                            ("gru_text", "candidate_hash"),
                            ("gru_centroids", "candidate_fingerprint")):
        if raw[model][key_name] != candidate_hash:
            raise ValueError(f"{model}: candidate fingerprint differs")
    result = {}
    for h in range(1, 6):
        key = str(h)
        choices = {
            "xgb_citations": raw["xgb_citations"]["selected_by_horizon"][key],
            "xgb_text": raw["xgb_text"],
            "xgb_centroids": raw["xgb_centroids"]["horizons"][key],
            "gru_citations": raw["gru_citations"]["settings_by_horizon"][key],
            "gru_text": raw["gru_text"]["settings_by_horizon"][key],
            "gru_centroids": raw["gru_centroids"]["selected_by_horizon"][key],
        }
        if choices["xgb_citations"]["data_hash"] != expected_hashes[h]:
            raise ValueError(f"H={h}: XGB citation ID hash mismatch")
        if choices["xgb_centroids"]["ids_hash"] != expected_hashes[h]:
            raise ValueError(f"H={h}: centroid ID hash mismatch")
        for model in ("gru_text", "gru_centroids"):
            if choices[model]["ids_hash"] != expected_hashes[h]:
                raise ValueError(f"H={h}: {model} ID hash mismatch")
        choice = {}
        for model in MODEL_NAMES:
            entry = choices[model]
            if model == "xgb_citations":
                number, params = entry["candidate"], entry["params"]
            elif model == "xgb_text":
                number, params = entry["candidate"], entry["params"]
            elif model == "xgb_centroids":
                number, params = entry["candidate"], entry["parameters"]
            else:
                config = entry["configuration"]
                number, params = config["candidate"], config
                if config != candidates[int(number)]:
                    raise ValueError(f"H={h}: {model} differs from candidate list")
            choice[model] = {"candidate": int(number), "params": params,
                             "params_hash": sha(params)}
        result[h] = choice
    if raw["xgb_text"]["text_dimensions"] != 384:
        raise ValueError("Unexpected XGB text dimension")
    if raw["gru_text"].get("encoder") != "frozen all-MiniLM-L6-v2":
        raise ValueError("Unexpected GRU text encoder")
    return result


def combined_matrix(features, embedding, h1_rows, indices):
    """Make an indexed semantic matrix with small temporary read blocks."""
    width = embedding.shape[1]
    result = np.empty((len(indices), 21 + width), dtype=np.float32)
    for start in range(0, len(indices), 4096):
        stop = min(start + 4096, len(indices))
        selected = indices[start:stop]
        result[start:stop, :21] = features[selected]
        result[start:stop, 21:] = embedding[h1_rows[selected]]
    return result


def predict_xgb(name, horizon_data, train, test, choice, text, centroids,
                shared, temporary_dir):
    from xgboost import XGBRegressor

    params = choice["params"]
    features = build_features(horizon_data["X"])
    h1_rows = horizon_data["h1_rows"]
    temp_path = None
    if name == "xgb_citations":
        xtrain = features[train]
    elif name == "xgb_text":
        xtrain = combined_matrix(features, text, h1_rows, train)
    else:
        if shared is None:
            raise FileNotFoundError("Original combined centroid feature file required")
        # A disk-backed array avoids a second full-size RAM copy of the
        # 1,045-column training matrix. XGBoost may still allocate RAM.
        needed = len(train) * shared.shape[1] * np.dtype(np.float32).itemsize
        if shutil.disk_usage(temporary_dir).free < 2 * needed:
            raise OSError("Insufficient free space for centroid training matrix")
        fd, name_on_disk = tempfile.mkstemp(
            prefix="centroid_fit_", suffix=".npy", dir=temporary_dir)
        os.close(fd)
        temp_path = Path(name_on_disk)
        xtrain = np.lib.format.open_memmap(
            temp_path, mode="w+", dtype=np.float32,
            shape=(len(train), shared.shape[1]))
        for start in range(0, len(train), 4096):
            stop = min(start + 4096, len(train))
            xtrain[start:stop] = shared[h1_rows[train[start:stop]]]
        xtrain.flush()
    model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        n_jobs=min(8, os.cpu_count() or 4),
        random_state=42 if name != "xgb_citations" else int(choice["seed"]),
        **params,
    )
    model.fit(xtrain, np.log1p(horizon_data["y"][train]))
    del xtrain
    gc.collect()
    if name == "xgb_citations":
        xtest = features[test]
    elif name == "xgb_text":
        xtest = combined_matrix(features, text, h1_rows, test)
    else:
        xtest = np.empty((len(test), shared.shape[1]), dtype=np.float32)
        for start in range(0, len(test), 4096):
            stop = min(start + 4096, len(test))
            xtest[start:stop] = shared[h1_rows[test[start:stop]]]
    predicted = model.predict(xtest).astype(np.float64)
    del model, xtest, features
    gc.collect()
    if temp_path is not None:
        temp_path.unlink()
    return np.clip(np.expm1(predicted), 0, None), {}


def gru_model_type():
    import torch
    from torch import nn

    class FullPanelCitationGRU(nn.Module):
        def __init__(self, config, semantic_dim):
            super().__init__()
            hidden = int(config["gru_hidden"])
            layers = int(config["gru_layers"])
            head = int(config["head_hidden"])
            dropout = float(config["dropout"])
            projection = int(config["semantic_projection"])
            self.semantic_dim = semantic_dim
            self.gru = nn.GRU(1, hidden, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0)
            if semantic_dim:
                self.semantic_branch = nn.Sequential(
                    nn.Linear(semantic_dim, projection), nn.ReLU(), nn.Dropout(dropout))
            fused = hidden + 4 + (projection if semantic_dim else 0)
            second = max(32, head // 2)
            self.prediction_head = nn.Sequential(
                nn.Linear(fused, head), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(head, second), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(second, 1))

        def forward(self, sequence, scalars, semantic=None):
            _, state = self.gru(sequence.unsqueeze(-1))
            parts = [state[-1], scalars]
            if self.semantic_dim:
                if semantic is None:
                    raise ValueError("Missing semantic input")
                parts.append(self.semantic_branch(semantic))
            return self.prediction_head(torch.cat(parts, dim=1)).squeeze(-1)

    return FullPanelCitationGRU


def predict_gru(name, horizon_data, train, test, choice, text, centroids,
                seed, max_epochs, patience):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    config = choice["params"]
    semantic = (None if name == "gru_citations" else
                text if name == "gru_text" else centroids)
    semantic_dim = 0 if semantic is None else semantic.shape[1]
    model_type = gru_model_type()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    sequence, scalars = build_gru_inputs(horizon_data["X"])
    target_log = np.log1p(horizon_data["y"]).astype(np.float32)
    fit, val = train_test_split(train, test_size=.20, random_state=seed)
    if np.intersect1d(np.concatenate((fit, val)), test).size:
        raise ValueError("GRU inner validation overlaps test")

    def new_model():
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        net = model_type(config, semantic_dim).to(device)
        opt = torch.optim.AdamW(net.parameters(),
                                lr=float(config["learning_rate"]),
                                weight_decay=float(config["weight_decay"]))
        return net, opt

    def make_loader(indices, seq_scale, scal_scale, shuffle):
        tensors = (
            torch.from_numpy(seq_scale.transform(sequence[indices]).astype(np.float32)),
            torch.from_numpy(scal_scale.transform(scalars[indices]).astype(np.float32)),
            torch.from_numpy(horizon_data["h1_rows"][indices].astype(np.int64)),
            torch.from_numpy(target_log[indices]),
        )
        return DataLoader(TensorDataset(*tensors), batch_size=int(config["batch_size"]),
                          shuffle=shuffle, num_workers=0,
                          generator=torch.Generator().manual_seed(seed))

    def call(net, sq, sc, rows):
        sq, sc = sq.to(device), sc.to(device)
        if semantic is None:
            return net(sq, sc)
        vectors = np.asarray(semantic[rows.numpy()], dtype=np.float32).copy()
        return net(sq, sc, torch.from_numpy(vectors).to(device))

    def one_epoch(net, opt, loader):
        net.train()
        for sq, sc, rows, truth in loader:
            opt.zero_grad()
            pred = call(net, sq, sc, rows)
            loss = nn.functional.mse_loss(pred, truth.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite GRU training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), float(config["gradient_clip"]))
            opt.step()

    seq_fit = StandardScaler().fit(sequence[fit])
    scal_fit = StandardScaler().fit(scalars[fit])
    fit_loader = make_loader(fit, seq_fit, scal_fit, True)
    val_loader = make_loader(val, seq_fit, scal_fit, False)
    net, opt = new_model()
    best_loss, best_epoch, bad = math.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        one_epoch(net, opt, fit_loader)
        net.eval()
        squared = 0.0
        with torch.inference_mode():
            for sq, sc, rows, truth in val_loader:
                error = call(net, sq, sc, rows).cpu().numpy() - truth.numpy()
                squared += float(np.square(error.astype(np.float64)).sum())
        score = squared / len(val)
        if not np.isfinite(score):
            raise ValueError("Non-finite GRU validation loss")
        if score < best_loss:
            best_loss, best_epoch, bad = score, epoch, 0
        else:
            bad += 1
        if bad >= patience:
            break
    del net, opt, fit_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Same two-stage policy as the final subset experiment: refit all outer
    # training patents for the selected epoch count, with fresh train-only scales.
    seq_train = StandardScaler().fit(sequence[train])
    scal_train = StandardScaler().fit(scalars[train])
    train_loader = make_loader(train, seq_train, scal_train, True)
    test_loader = make_loader(test, seq_train, scal_train, False)
    net, opt = new_model()
    for _ in range(best_epoch):
        one_epoch(net, opt, train_loader)
    net.eval()
    output = []
    with torch.inference_mode():
        for sq, sc, rows, _ in test_loader:
            output.append(call(net, sq, sc, rows).cpu().numpy())
    predicted = np.concatenate(output).astype(np.float64)
    del net, opt, train_loader, test_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.clip(np.expm1(predicted), 0, None), {
        "best_epoch": best_epoch, "validation_log_MSE": best_loss,
    }


def evaluate(y, predictions, years):
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("Non-finite or negative predictions")
    scores = dict(MAE=float(mean_absolute_error(y, predictions)),
                  rho=float(spearmanr(y, predictions).statistic),
                  p10=p10_pooled(y, predictions),
                  p10_coh=p10_cohort(y, predictions, years))
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("Invalid test score")
    return scores


def verify_or_save(path, horizon, seed, name, choice, d, test,
                   predictions=None, diagnostics=None):
    ids, years, target = d["ids"][test], d["yr"][test], d["y"][test]
    h = ids_sha(d["ids"])
    if path.exists():
        saved = pd.read_csv(path)
        required = {"record_id", "pub_year", "y_true", "y_pred", "H", "seed",
                    "model", "candidate", "ids_hash", "params_hash"}
        if not required.issubset(saved.columns):
            raise ValueError(f"Incomplete saved predictions: {path}")
        if name.startswith("gru_") and not {
            "best_epoch", "validation_log_MSE"}.issubset(saved.columns):
            raise ValueError(f"Missing GRU epoch diagnostics: {path}")
        if (not np.array_equal(saved.record_id.to_numpy(dtype=np.int64), ids)
                or not np.array_equal(saved.pub_year.to_numpy(), years)
                or not np.array_equal(saved.y_true.to_numpy(), target)
                or not (saved.H == horizon).all() or not (saved.seed == seed).all()
                or not (saved.model == name).all()
                or not (saved.candidate == choice["candidate"]).all()
                or not (saved.ids_hash == h).all()
                or not (saved.params_hash == choice["params_hash"]).all()):
            raise ValueError(f"Saved predictions mismatch current inputs: {path}")
        return saved.y_pred.to_numpy(dtype=np.float64), saved
    if predictions is None:
        return None, None
    saved = pd.DataFrame(dict(record_id=ids, pub_year=years, y_true=target,
                              y_pred=predictions, H=horizon, seed=seed, model=name,
                              candidate=choice["candidate"], ids_hash=h,
                              params_hash=choice["params_hash"]))
    if name.startswith("gru_"):
        if not diagnostics or not all(k in diagnostics for k in (
                "best_epoch", "validation_log_MSE")):
            raise ValueError("GRU epoch diagnostics were not supplied")
        saved["best_epoch"] = diagnostics["best_epoch"]
        saved["validation_log_MSE"] = diagnostics["validation_log_MSE"]
    atomic_csv(saved, path)
    return predictions, saved


def summarize(results, out, seeds):
    frame = pd.DataFrame(results).sort_values(["H", "model", "seed"])
    atomic_csv(frame, out / "metrics_per_split.csv")
    counts = frame.groupby(["H", "model"]).seed.nunique()
    if not (counts == len(seeds)).all() or len(frame) != 5 * 6 * len(seeds):
        print("Partial results saved; complete summary will be written after all fits")
        return
    metrics = ["MAE", "rho", "p10", "p10_coh"]
    summary = frame.groupby(["H", "model"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    atomic_csv(summary.reset_index(), out / "summary_mean_sd.csv")
    comparisons = []
    for h in range(1, 6):
        for family in ("xgb", "gru"):
            base = frame[(frame.H == h) & (frame.model == f"{family}_citations")]
            for extra in ("text", "centroids"):
                new = frame[(frame.H == h) & (frame.model == f"{family}_{extra}")]
                pair = base.merge(new, on=["H", "seed"], suffixes=("_base", "_extra"),
                                  validate="one_to_one")
                for _, row in pair.iterrows():
                    comparisons.append(dict(H=h, family=family, extra=extra,
                                            seed=int(row.seed),
                                            MAE_gain=row.MAE_base-row.MAE_extra,
                                            rho_gain=row.rho_extra-row.rho_base,
                                            p10_gain=row.p10_extra-row.p10_base,
                                            p10_coh_gain=row.p10_coh_extra-row.p10_coh_base))
    atomic_csv(pd.DataFrame(comparisons), out / "paired_gains_per_split.csv")
    print("Final 20-split summary and paired gains saved", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("final20"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-stop", type=int, default=40, help="Exclusive")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                        default=list(MODEL_NAMES))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(range(1, 6)))
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    if args.seed_start < 0 or args.seed_stop <= args.seed_start:
        raise ValueError("Invalid seed interval")
    if any(h not in SIZES for h in args.horizons):
        raise ValueError("Horizon must be 1--5")
    if not args.dry_run and (args.seed_start, args.seed_stop) != (20, 40):
        print("WARNING: nonstandard seed range; outputs will be partial", flush=True)
    data_dir = args.data_dir.resolve()
    out = args.output_dir.resolve()
    h1_ids = pd.read_csv(data_dir / "patent_text_prepared_H1.csv",
                         usecols=["record_id"], dtype={"record_id": np.int64}).record_id.to_numpy()
    if len(h1_ids) != SIZES[1] or len(np.unique(h1_ids)) != len(h1_ids):
        raise ValueError("Invalid H=1 prepared text IDs")
    selected_cit = read_json(data_dir / "fullpanel_xgb_citations_selected_settings_H1_to_H5.json")
    expected_hashes = {h: selected_cit["selected_by_horizon"][str(h)]["data_hash"]
                       for h in range(1, 6)}
    if ids_sha(h1_ids) != expected_hashes[1]:
        raise ValueError("H=1 text IDs differ from tuning")
    data = load_panel(data_dir, h1_ids, expected_hashes)
    choices = load_settings(data_dir, expected_hashes)
    text = np.load(data_dir / "fullpanel_H1_MiniLM_frozen.npy", mmap_mode="r")
    centroids = np.load(data_dir / "fullpanel_H1_centroids_original_pooled.npy", mmap_mode="r")
    centroid_ids = np.load(data_dir / "fullpanel_H1_centroids_original_record_ids.npy")
    if text.shape != (len(h1_ids), 384) or centroids.shape != (len(h1_ids), 1024):
        raise ValueError("Unexpected semantic embedding dimensions")
    if not np.array_equal(centroid_ids, h1_ids):
        raise ValueError("Centroid row IDs differ from H=1")
    shared_file = data_dir / "fullpanel_H1_xgb_original_centroid_features.npy"
    shared = np.load(shared_file, mmap_mode="r") if shared_file.is_file() else None
    if "xgb_centroids" in args.models and (shared is None or shared.shape != (len(h1_ids), 1045)):
        raise ValueError("Missing or invalid 1,045-column centroid XGBoost file")
    if shared is not None:
        sample = np.array([0, len(h1_ids) // 2, len(h1_ids) - 1])
        if (not np.array_equal(shared[sample, :21], build_features(data[1]["X"][sample]))
                or not np.array_equal(shared[sample, 21:], centroids[sample])):
            raise ValueError("Combined centroid features differ from tuning inputs")
    if args.dry_run:
        for h in range(1, 6):
            if not np.array_equal(h1_ids[data[h]["h1_rows"]], data[h]["ids"]):
                raise ValueError(f"H={h}: semantic row alignment failed")
            seq, scal = build_gru_inputs(data[h]["X"][:8])
            if seq.shape != (8, 5) or scal.shape != (8, 4):
                raise ValueError(f"H={h}: GRU input shapes failed")
        if any(m.startswith("gru_") for m in args.models):
            import torch
            cls = gru_model_type()
            for name, dim in (("gru_citations", 0), ("gru_text", 384),
                              ("gru_centroids", 1024)):
                config = choices[5][name]["params"]
                net = cls(config, dim)
                sq, sc = build_gru_inputs(data[5]["X"][:8])
                semantic = None if dim == 0 else (text if dim == 384 else centroids)
                with torch.no_grad():
                    vec = None if semantic is None else torch.from_numpy(
                        np.asarray(semantic[data[5]["h1_rows"][:8]]).copy())
                    result = net(torch.from_numpy(sq), torch.from_numpy(sc), vec)
                if result.shape != (8,):
                    raise ValueError(f"{name}: forward pass failed")
        print("Dry run passed: inputs/configs/architecture aligned; no model fitted", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seed_start, args.seed_stop)
    results = []
    for h in args.horizons:
        d = data[h]
        for seed in seeds:
            train, test = train_test_split(np.arange(len(d["ids"])),
                                           test_size=.20, random_state=seed)
            if np.intersect1d(train, test).size:
                raise ValueError("Outer split overlaps")
            for name in args.models:
                choice = dict(choices[h][name], seed=seed)
                path = out / f"H{h}_seed{seed}_{name}_predictions.csv"
                predicted, saved = verify_or_save(path, h, seed, name, choice, d, test)
                diagnostics = {}
                if predicted is not None and name.startswith("gru_"):
                    diagnostics = {
                        "best_epoch": int(saved.best_epoch.iloc[0]),
                        "validation_log_MSE": float(saved.validation_log_MSE.iloc[0]),
                    }
                if predicted is None:
                    print(f"H={h} seed={seed} {name}: fitting", flush=True)
                    if name.startswith("xgb_"):
                        predicted, diagnostics = predict_xgb(
                            name, d, train, test, choice, text, centroids,
                            shared, out)
                    else:
                        predicted, diagnostics = predict_gru(
                            name, d, train, test, choice, text, centroids,
                            seed, args.max_epochs, args.patience)
                    predicted, _ = verify_or_save(path, h, seed, name, choice,
                                                   d, test, predicted, diagnostics)
                score = evaluate(d["y"][test], predicted, d["yr"][test])
                row = dict(H=h, seed=seed, model=name, candidate=choice["candidate"],
                           test_patents=len(test), **score, **diagnostics)
                results.append(row)
                atomic_csv(pd.DataFrame(results), out / "metrics_per_split.csv")
                print(f"H={h} seed={seed} {name}: complete, P@10={score['p10']:.4f}",
                      flush=True)
        gc.collect()
    if (len(args.models) == 6 and len(args.horizons) == 5
            and (args.seed_start, args.seed_stop) == (20, 40)):
        summarize(results, out, seeds)
    else:
        print("Requested subset completed; complete summary requires all six models and five horizons")


if __name__ == "__main__":
    main()
