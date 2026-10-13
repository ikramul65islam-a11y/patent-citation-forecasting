"""Complete, resumable replacement XGBoost evaluation for six input families.

Run in the Jupyter working folder holding the prepared inputs and selections.
Writes ONLY final_six_xgb_rerun_v1/. Existing results remain untouched.
Requires numpy, pandas, scipy, scikit-learn, xgboost.
"""
import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from scipy.stats import spearmanr, t as student_t
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, XGBRegressor

OUT = Path("final_six_xgb_rerun_v1")
OUT.mkdir(exist_ok=True)
SEEDS = tuple(range(20, 40))
FAMILIES = ("citations", "text", "centroids", "paper", "finance", "combined")
EXPECTED = {
    1: (268192, "d0942b5bd4cf", 53639),
    2: (248874, "6a8ee6fd7cab", 49775),
    3: (225128, "a1065924d755", 45026),
    4: (201400, "72c6ac5f90a2", 40280),
    5: (178765, "2a3a91b3a7f8", 35753),
}
SETTINGS = {
    "citations": "fullpanel_xgb_citations_selected_settings_H1_to_H5.json",
    "text": "fullpanel_xgb_text_selected_settings.json",
    "centroids": "fullpanel_xgb_centroids_original_selected_settings_H1_to_H5.json",
    "paper": "fullpanel_xgb_paper_selected_H1_to_H5.json",
    "finance": "fullpanel_xgb_finance_selected_H1_to_H5.json",
    "combined": "fullpanel_xgb_paper_finance_selected_H1_to_H5.json",
}
EXPECTED_SOURCE_CANDIDATES = {
    "paper": (4, 5, 4, 5, 1),
    "finance": (4, 1, 4, 1, 1),
    "combined": (4, 5, 1, 5, 1),
}
PARAM_NAMES = {
    "n_estimators", "max_depth", "learning_rate", "min_child_weight",
    "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
}
INPUT_FILES = {
    "source_features": "fullpanel_scientific_market_features_H1.npz",
    "citation_panel": "fullpanel_citation_age0_9_H1_aligned.npz",
    "text": "fullpanel_H1_MiniLM_frozen.npy",
    "centroids": "fullpanel_H1_xgb_original_centroid_features.npy",
    "centroid_ids": "fullpanel_H1_centroids_original_record_ids.npy",
}


def file_hash(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json_atomic(data, path):
    temporary = str(path) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True, indent=2)
    os.replace(temporary, path)


def metrics(y, prediction, years):
    if (len(y) != len(prediction) or len(y) != len(years)
            or not np.isfinite(prediction).all() or (prediction < 0).any()):
        raise ValueError("Invalid predictions or alignment")
    k = int(len(y) * 0.10)
    p10 = float((y[np.argsort(-prediction)[:k]] >= np.quantile(y, .90)).mean())
    hits = selected = 0
    for year in np.unique(years):
        mask = years == year
        if mask.sum() < 20:
            continue
        count = max(1, int(mask.sum() * .10))
        yc, pc = y[mask], prediction[mask]
        hits += int((yc[np.argsort(-pc)[:count]] >= np.quantile(yc, .90)).sum())
        selected += count
    if not selected:
        raise ValueError("No sufficiently large publication cohort")
    return {"MAE": float(np.mean(np.abs(y - prediction))),
            "rho": float(spearmanr(y, prediction).statistic),
            "p10": p10, "p10_coh": float(hits / selected)}


for label, filename in {**INPUT_FILES, **SETTINGS}.items():
    if not Path(filename).is_file():
        raise FileNotFoundError(f"{label}: {filename}")

inputs_sha = {name: file_hash(filename) for name, filename in INPUT_FILES.items()}
settings_sha = {name: file_hash(filename) for name, filename in SETTINGS.items()}
selected = {}
for family, filename in SETTINGS.items():
    with open(filename, encoding="utf-8") as handle:
        selected[family] = json.load(handle)

with np.load(INPUT_FILES["source_features"], allow_pickle=False) as saved:
    ids_h1, years_h1 = saved["record_id"].copy(), saved["pub_year"].copy()
    base, paper, finance, covered = (saved[key].copy() for key in
                                     ("citation_21", "paper_42", "finance_21",
                                      "source_years_covered"))
    base_names = saved["citation_names"].astype(str).tolist()
    paper_names = saved["paper_names"].astype(str).tolist()
    finance_names = saved["finance_names"].astype(str).tolist()
with np.load(INPUT_FILES["citation_panel"], allow_pickle=False) as saved:
    if (not np.array_equal(ids_h1, saved["record_id"])
            or not np.array_equal(years_h1, saved["pub_year"])):
        raise ValueError("Citation rows and source feature rows differ")
    history = saved["citation_age0_9"].copy()
    observed = saved["citation_observed_age0_9"].copy()
if (ids_h1.shape != (268192,) or base.shape != (268192, 21)
        or paper.shape != (268192, 42) or finance.shape != (268192, 21)
        or covered.shape != (268192, 1) or history.shape != (268192, 10)
        or observed.shape != history.shape or not observed[:, :5].all()):
    raise ValueError("Unexpected input dimensions or missing input ages")
text = np.load(INPUT_FILES["text"], mmap_mode="r")
centroids = np.load(INPUT_FILES["centroids"], mmap_mode="r")
centroid_ids = np.load(INPUT_FILES["centroid_ids"])
if (text.shape != (268192, 384) or centroids.shape != (268192, 1045)
        or not np.array_equal(centroid_ids, ids_h1)
        or not np.array_equal(centroids[:, :21], base)):
    raise ValueError("Text/centroid dimensions, IDs, or citation features differ")

source_matrices = {
    "paper": np.column_stack((base, paper, covered)).astype(np.float32),
    "finance": np.column_stack((base, finance, covered)).astype(np.float32),
    "combined": np.column_stack((base, paper, finance, covered)).astype(np.float32),
}
feature_names = {
    "citations": base_names,
    "text": base_names + [f"MiniLM_{i:03d}" for i in range(384)],
    "centroids": base_names + [f"Centroid_{i:04d}" for i in range(1024)],
    "paper": base_names + paper_names + ["source_years_covered"],
    "finance": base_names + finance_names + ["source_years_covered"],
    "combined": base_names + paper_names + finance_names + ["source_years_covered"],
}
groups = {
    "citations": [("Early citations", 0, 21)],
    "text": [("Early citations", 0, 21), ("Patent text", 21, 405)],
    "centroids": [("Early citations", 0, 21), ("Centroids", 21, 1045)],
    "paper": [("Early citations", 0, 21), ("Paper activity", 21, 63),
              ("Coverage", 63, 64)],
    "finance": [("Early citations", 0, 21), ("Financial activity", 21, 42),
                ("Coverage", 42, 43)],
    "combined": [("Early citations", 0, 21), ("Paper activity", 21, 63),
                 ("Financial activity", 63, 84), ("Coverage", 84, 85)],
}
if any(source_matrices[name].shape[1] != len(feature_names[name])
       for name in source_matrices):
    raise ValueError("Source feature names or widths differ")


def selected_setting(family, horizon, id_hash):
    root = selected[family]
    key = str(horizon)
    if family == "citations":
        entry = root["selected_by_horizon"][key]
        if entry["data_hash"] != id_hash:
            raise ValueError(f"H={horizon}: citation tuning ID mismatch")
        candidate, params = entry["candidate"], entry["params"]
    elif family == "text":
        if root["text_dimensions"] != 384 or root["candidate"] != 5:
            raise ValueError("Text selection or representation differs")
        candidate, params = root["candidate"], root["params"]
    elif family == "centroids":
        entry = root["horizons"][key]
        if entry["ids_hash"] != id_hash or entry["centroid_dimensions"] != 1024:
            raise ValueError(f"H={horizon}: centroid tuning sample differs")
        candidate, params = entry["candidate"], entry["parameters"]
    else:
        if (root["feature_sha256"] != inputs_sha["source_features"]
                or root["citation_sha256"] != inputs_sha["citation_panel"]
                or root["validation_seeds"] != [100, 101, 102, 103, 104]
                or root["reserved_test_used_for_selection"] is not False):
            raise ValueError(f"{family}: tuning provenance differs")
        entry = root["settings_by_horizon"][key]
        if entry["ids_hash"] != id_hash:
            raise ValueError(f"H={horizon}: {family} tuning sample differs")
        candidate, params = entry["candidate"], entry["params"]
        if candidate != EXPECTED_SOURCE_CANDIDATES[family][horizon - 1]:
            raise ValueError(f"H={horizon}: {family} selected candidate changed")
    if set(params) != PARAM_NAMES:
        raise ValueError(f"{family} H={horizon}: selected parameters incomplete")
    return int(candidate), params


datasets = {}
choices = {}
for H, (count, prefix, n_test) in EXPECTED.items():
    mask = observed[:, :5 + H].all(axis=1)
    h1_positions = np.flatnonzero(mask)
    ids = ids_h1[mask]
    id_hash = hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest()
    if len(ids) != count or not id_hash.startswith(prefix):
        raise ValueError(f"H={H}: patent sample differs")
    y = history[mask, 5:5 + H].sum(axis=1, dtype=np.int64)
    if (y < 0).any():
        raise ValueError(f"H={H}: negative target counts")
    datasets[H] = (h1_positions, ids, years_h1[mask], y, id_hash, n_test)
    choices[H] = {name: selected_setting(name, H, id_hash) for name in FAMILIES}

protocol = {
    "label": "six_locked_xgb_seed20_to39_v1",
    "seeds": list(SEEDS), "horizons": list(EXPECTED), "families": list(FAMILIES),
    "input_sha256": inputs_sha, "settings_sha256": settings_sha,
    "candidate_by_horizon": {str(H): {name: choices[H][name][0]
                                     for name in FAMILIES} for H in EXPECTED},
    "xgboost_version": xgboost.__version__,
    "fit": {"objective": "reg:squarederror", "tree_method": "hist",
            "n_jobs": min(8, os.cpu_count() or 4),
            "random_state": "split_seed for citations; 42 otherwise",
            "target": "float32 log1p(sum of citations at ages 5 through 4+H)"},
    "split": "sklearn train_test_split(test_size=.20, random_state=split_seed)",
    "metrics": ["MAE", "rho", "p10", "p10_coh"],
    "shap": "seed20, 2000 held-out records, Tree SHAP log prediction scale",
}
run_hash = stable_hash(protocol)
manifest = OUT / "locked_run_manifest.json"
if manifest.is_file():
    with open(manifest, encoding="utf-8") as handle:
        if json.load(handle) != protocol:
            raise ValueError("Existing output folder uses another protocol or environment")
else:
    write_json_atomic(protocol, manifest)
print("Locked run hash:", run_hash)
print("XGBoost:", xgboost.__version__)
print("Saved settings and inputs checked; no old test predictions read.")


def matrix_for(family, h1_positions, train, test):
    """Return fit/test matrices and optional disk-backed fit filename."""
    temp = None
    if family in source_matrices:
        source = source_matrices[family]
        xtrain = source[h1_positions[train]]
        xtest = source[h1_positions[test]]
    elif family == "citations":
        xtrain = base[h1_positions[train]]
        xtest = base[h1_positions[test]]
    elif family == "text":
        xtrain = np.column_stack((base[h1_positions[train]],
                                  text[h1_positions[train]])).astype(np.float32)
        xtest = np.column_stack((base[h1_positions[test]],
                                 text[h1_positions[test]])).astype(np.float32)
    else:
        descriptor, temp = tempfile.mkstemp(
            prefix="centroid_fit_", suffix=".npy", dir=OUT)
        os.close(descriptor)
        xtrain = np.lib.format.open_memmap(
            temp, mode="w+", dtype=np.float32, shape=(len(train), 1045))
        for start in range(0, len(train), 4096):
            stop = min(start + 4096, len(train))
            xtrain[start:stop] = centroids[h1_positions[train[start:stop]]]
        xtrain.flush()
        xtest = np.empty((len(test), 1045), dtype=np.float32)
        for start in range(0, len(test), 4096):
            stop = min(start + 4096, len(test))
            xtest[start:stop] = centroids[h1_positions[test[start:stop]]]
    return xtrain, xtest, temp


def test_matrix_for(family, h1_positions, test):
    positions = h1_positions[test]
    if family in source_matrices:
        return source_matrices[family][positions]
    if family == "citations":
        return base[positions]
    if family == "text":
        return np.column_stack((base[positions], text[positions])).astype(np.float32)
    return np.asarray(centroids[positions], dtype=np.float32)


def save_shap(booster, xtest, ids, test, H, family, stem):
    path = stem.with_name(stem.name + "_shap.npz")
    group_path = stem.with_name(stem.name + "_groups.csv")
    feature_path = stem.with_name(stem.name + "_features.csv")
    if all(p.is_file() for p in (path, group_path, feature_path)):
        with np.load(path, allow_pickle=False) as existing:
            sample = np.sort(np.random.default_rng(2026 + H).choice(
                len(test), 2000, replace=False))
            if (not np.array_equal(existing["record_id"], ids[test[sample]])
                    or str(existing["run_hash"].item()) != run_hash):
                raise ValueError(f"{path}: saved SHAP sample/protocol differs")
        return
    sample = np.sort(np.random.default_rng(2026 + H).choice(
        len(test), 2000, replace=False))
    sampled_X = np.asarray(xtest[sample]).copy()
    dm = DMatrix(sampled_X)
    contributions = booster.predict(dm, pred_contribs=True)
    margin = booster.predict(dm, output_margin=True)
    if (contributions.shape != (2000, len(feature_names[family]) + 1)
            or np.max(np.abs(contributions.sum(axis=1) - margin)) > 1e-4):
        raise ValueError(f"{stem.name}: Tree SHAP additivity failed")
    shap_values = contributions[:, :-1]
    temporary = str(path) + ".tmp.npz"
    np.savez_compressed(temporary, record_id=ids[test[sample]],
                        X_sample=sampled_X, shap_values=shap_values,
                        base_values=contributions[:, -1],
                        feature_names=np.asarray(feature_names[family], dtype="U64"),
                        run_hash=np.array(run_hash))
    os.replace(temporary, path)
    pd.DataFrame({"feature": feature_names[family],
                  "mean_abs_SHAP_log_scale": np.abs(shap_values).mean(axis=0)}).sort_values(
                      "mean_abs_SHAP_log_scale", ascending=False).to_csv(feature_path, index=False)
    pd.DataFrame([{"group": label,
                   "mean_abs_group_SHAP_log_scale":
                       float(np.abs(shap_values[:, left:right].sum(axis=1)).mean())}
                  for label, left, right in groups[family]]).to_csv(group_path, index=False)


for H, (h1_positions, ids, years, y, id_hash, n_test) in datasets.items():
    for seed in SEEDS:
        train, test = train_test_split(
            np.arange(len(ids)), test_size=.20, random_state=seed)
        if len(test) != n_test or np.intersect1d(train, test).size:
            raise ValueError(f"H={H}, seed={seed}: split differs")
        split_hash = hashlib.sha256(np.concatenate((train, test))
                                    .astype(np.int64).tobytes()).hexdigest()
        for family in FAMILIES:
            candidate, params = choices[H][family]
            stem = OUT / f"{family}_H{H}_seed{seed}"
            prediction_path = stem.with_name(stem.name + "_predictions.npz")
            model_path = stem.with_name(stem.name + "_model.ubj")
            expected_meta = {"H": H, "seed": seed, "family": family,
                             "candidate": candidate, "params": params,
                             "ids_hash": id_hash, "split_hash": split_hash,
                             "run_hash": run_hash, "xgboost_version": xgboost.__version__}
            if prediction_path.is_file() and model_path.is_file():
                with np.load(prediction_path, allow_pickle=False) as saved:
                    meta = json.loads(str(saved["metadata_json"].item()))
                    if (any(meta.get(key) != value for key, value in expected_meta.items()) or
                            meta.get("model_sha256") != file_hash(model_path) or
                            not np.array_equal(saved["record_id"], ids[test]) or
                            not np.array_equal(saved["pub_year"], years[test]) or
                            not np.array_equal(saved["y_true"], y[test])):
                        raise ValueError(f"{prediction_path}: checkpoint differs")
                    old_pred = saved["y_pred"].copy()
                booster = xgboost.Booster()
                booster.load_model(str(model_path))
                # Match model predictions to the saved checkpoint on selected rows.
                checked_X = test_matrix_for(family, h1_positions, test[:16])
                checked = booster.predict(DMatrix(checked_X))
                count_checked = np.clip(np.expm1(
                    checked if family in source_matrices else checked.astype(np.float64)),
                    0, None)
                if not np.allclose(count_checked, old_pred[:16], atol=1e-5, rtol=1e-6):
                    raise ValueError(f"{model_path}: model and predictions differ")
                del checked_X
                if seed == 20:
                    xtest = test_matrix_for(family, h1_positions, test)
                    save_shap(booster, xtest, ids, test, H, family, stem)
                    del xtest
                print(f"H={H} seed={seed} {family}: checkpoint verified", flush=True)
                del booster
                gc.collect()
                continue
            temp = None
            try:
                xtrain, xtest, temp = matrix_for(family, h1_positions, train, test)
                model = XGBRegressor(
                    objective="reg:squarederror", tree_method="hist",
                    n_jobs=min(8, os.cpu_count() or 4),
                    random_state=seed if family == "citations" else 42,
                    **params)
                model.fit(xtrain, np.log1p(y[train].astype(np.float32)))
                del xtrain
                gc.collect()
                raw = model.predict(xtest)
                prediction = np.clip(np.expm1(
                    raw if family in source_matrices else raw.astype(np.float64)),
                    0, None)
                if family in source_matrices:
                    prediction = prediction.astype(np.float32)
                scores = metrics(y[test], prediction, years[test])
                tmp_model = str(model_path) + ".tmp.ubj"
                model.save_model(tmp_model)
                os.replace(tmp_model, model_path)
                saved_meta = {**expected_meta, "model_sha256": file_hash(model_path)}
                tmp_pred = str(prediction_path) + ".tmp.npz"
                np.savez_compressed(tmp_pred, record_id=ids[test], pub_year=years[test],
                                    y_true=y[test], y_pred=prediction,
                                    metadata_json=np.array(json.dumps(saved_meta, sort_keys=True)))
                os.replace(tmp_pred, prediction_path)
                if seed == 20:
                    save_shap(model.get_booster(), xtest, ids, test, H, family, stem)
                print(f"H={H} seed={seed} {family}: saved; MAE={scores['MAE']:.4f}",
                      flush=True)
                del model, prediction, raw, xtest
            finally:
                gc.collect()
                if temp and os.path.isfile(temp): os.remove(temp)
    print(f"H={H}: all 20 splits and six families saved", flush=True)


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    indices = np.argsort(pvalues)
    ordered = pvalues[indices]
    adjusted = np.maximum.accumulate(np.minimum(1., ordered *
                                                np.arange(len(ordered), 0, -1)))
    result = np.empty(len(adjusted))
    result[indices] = adjusted
    return result


per_split = []
comparisons = []
for H, (_, ids, years, y, _, n_test) in datasets.items():
    for seed in SEEDS:
        _, test = train_test_split(np.arange(len(ids)), test_size=.20,
                                   random_state=seed)
        pred = {}
        score = {}
        for family in FAMILIES:
            path = OUT / f"{family}_H{H}_seed{seed}_predictions.npz"
            with np.load(path, allow_pickle=False) as saved:
                if (not np.array_equal(saved["record_id"], ids[test]) or
                        not np.array_equal(saved["y_true"], y[test])):
                    raise ValueError(f"{path}: unmatched test patents")
                pred[family] = saved["y_pred"].copy()
            score[family] = metrics(y[test], pred[family], years[test])
            per_split.append({"H": H, "seed": seed, "family": family,
                              "test_patents": n_test, **score[family]})
        for family in FAMILIES[1:]:
            for metric in ("p10", "MAE", "rho", "p10_coh"):
                gain = (score["citations"][metric] - score[family][metric]
                        if metric == "MAE" else
                        score[family][metric] - score["citations"][metric])
                comparisons.append({"H": H, "seed": seed, "family": family,
                                    "metric": metric, "paired_gain": gain})

per_split = pd.DataFrame(per_split).sort_values(["H", "family", "seed"])
if (len(per_split) != 600 or not per_split.groupby(["H", "family"]).size().eq(20).all()):
    raise ValueError("Incomplete final rerun: refusing to calculate summary")
per_split.to_csv(OUT / "per_split_metrics.csv", index=False)
summary = (per_split.groupby(["H", "family"])[["MAE", "rho", "p10", "p10_coh"]]
           .agg(["mean", "std"]))
summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
summary.to_csv(OUT / "twenty_split_summary.csv")
paired = pd.DataFrame(comparisons)
paired.to_csv(OUT / "paired_gains_per_split.csv", index=False)

inference = []
for (family, H, metric), group in paired.groupby(["family", "H", "metric"]):
    values = group.sort_values("seed").paired_gain.to_numpy()
    n = len(values)
    train_n = EXPECTED[H][0] - EXPECTED[H][2]
    ratio = EXPECTED[H][2] / train_n
    corrected_se = values.std(ddof=1) * np.sqrt(1 / n + ratio)
    mean = float(values.mean())
    if corrected_se == 0:
        p = 1. if mean == 0 else 0.
        lower = upper = mean
    else:
        statistic = mean / corrected_se
        p = float(2 * student_t.sf(abs(statistic), df=n - 1))
        margin = float(student_t.ppf(.975, df=n - 1) * corrected_se)
        lower, upper = mean - margin, mean + margin
    inference.append({"family": family, "H": H, "metric": metric,
                      "mean_paired_gain": mean, "CI_95_low": lower,
                      "CI_95_high": upper, "corrected_p_two_sided": p,
                      "test_train_ratio": ratio, "n_splits": n})
inference = pd.DataFrame(inference)
inference["Holm_p"] = np.nan
inference["Holm_family"] = ""
for metric in ("p10", "MAE", "rho", "p10_coh"):
    for label, family_set in (("semantic_10", ("text", "centroids")),
                              ("external_15", ("paper", "finance", "combined"))):
        mask = inference.metric.eq(metric) & inference.family.isin(family_set)
        inference.loc[mask, "Holm_p"] = holm_adjust(
            inference.loc[mask, "corrected_p_two_sided"].to_numpy())
        inference.loc[mask, "Holm_family"] = label
inference.sort_values(["metric", "family", "H"]).to_csv(
    OUT / "paired_corrected_tests.csv", index=False)

print("\nFinal replacement 20-split XGBoost means and SD:")
print(summary.round(4).to_string())
print("\nPrimary P@10 comparisons, corrected repeated-split tests and Holm:")
print(inference.loc[inference.metric.eq("p10"),
                    ["family", "H", "mean_paired_gain", "CI_95_low",
                     "CI_95_high", "corrected_p_two_sided", "Holm_family",
                     "Holm_p"]].sort_values(["family", "H"]).to_string(index=False))
print("\nSaved everything in:", OUT)
