"""Run seed-20 Tree SHAP for six XGBoost families at H=1..5.

Paste this whole file into one Jupyter cell. The saved final predictions are
read only. All fitted models and explanations go to shap_six_xgb_refits/.
"""
import gc
import hashlib
import json
import os
import tempfile

import numpy as np
import pandas as pd
import xgboost
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, XGBRegressor

OUT = "shap_six_xgb_refits"
os.makedirs(OUT, exist_ok=True)
SIZES = (268192, 248874, 225128, 201400, 178765)
SOURCE = {"paper": "paper", "finance": "finance", "combined": "paper_finance"}
SELECTION = {
    "citations": "fullpanel_xgb_citations_selected_settings_H1_to_H5.json",
    "text": "fullpanel_xgb_text_selected_settings.json",
    "centroids": "fullpanel_xgb_centroids_original_selected_settings_H1_to_H5.json",
    **{k: f"fullpanel_xgb_{v}_selected_H1_to_H5.json" for k, v in SOURCE.items()},
}

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def score(y, pred):
    k = int(.1 * len(y))
    return {"MAE": float(np.mean(np.abs(y - pred))),
            "rho": float(spearmanr(y, pred).statistic),
            "p10": float(np.mean(y[np.argsort(-pred)[:k]] >= np.quantile(y, .9)))}

def saved_predictions(family, H, ids, years, y, test, candidate, params):
    if family in SOURCE:
        path = f"final_{SOURCE[family]}_xgb/H{H}_seed20_predictions.npz"
        with np.load(path, allow_pickle=False) as f:
            si, sy, st, sp = (f[k].copy() for k in
                               ("record_id", "pub_year", "y_true", "y_pred"))
            meta = json.loads(str(f["metadata_json"].item()))
        if (meta["H"] != H or meta["split_seed"] != 20
                or meta["candidate"] != candidate or meta["params"] != params
                or meta["ids_hash"] != hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest()
                or meta["feature_sha256"] != input_hash["feature"]
                or meta["citation_sha256"] != input_hash["citation"]
                or meta["settings_sha256"] != sha_file(SELECTION[family])):
            raise ValueError(f"{path}: original fit metadata differs")
    else:
        aliases = {"citations": ("xgb_citations",),
                   "text": ("xgb_text",),
                   "centroids": ("xgb_centroids_original", "xgb_centroids")}[family]
        options = [p for a in aliases for p in
                   (f"final20/{a}_H{H}_seed20.csv",
                    f"final20/H{H}_seed20_{a}_predictions.csv")]
        path = next((p for p in options if os.path.isfile(p)), None)
        if path is None:
            raise FileNotFoundError(options)
        f = pd.read_csv(path)
        si, sy, st, sp = (f[k].to_numpy() for k in
                           ("record_id", "pub_year", "y_true", "y_pred"))
        if (not f["candidate"].eq(candidate).all()
                or not f["H"].eq(H).all() or not f["seed"].eq(20).all()):
            raise ValueError(f"{path}: final run selection differs")
    if (not np.array_equal(si, ids[test]) or not np.array_equal(sy, years[test])
            or not np.array_equal(st, y[test]) or not np.isfinite(sp).all()):
        raise ValueError(f"{path}: final test patents or outcomes differ")
    return sp, path

with np.load("fullpanel_scientific_market_features_H1.npz", allow_pickle=False) as f:
    ids_h1, years_h1 = f["record_id"].copy(), f["pub_year"].copy()
    base, paper, finance, covered = (f[k].copy() for k in
                                     ("citation_21", "paper_42", "finance_21",
                                      "source_years_covered"))
    names_base = f["citation_names"].astype(str).tolist()
    names_paper = f["paper_names"].astype(str).tolist()
    names_finance = f["finance_names"].astype(str).tolist()
with np.load("fullpanel_citation_age0_9_H1_aligned.npz", allow_pickle=False) as f:
    if (not np.array_equal(ids_h1, f["record_id"])
            or not np.array_equal(years_h1, f["pub_year"])):
        raise ValueError("Feature and citation rows differ")
    history, observed = (f[k].copy() for k in
                         ("citation_age0_9", "citation_observed_age0_9"))
if (base.shape != (SIZES[0], 21) or paper.shape != (SIZES[0], 42)
        or finance.shape != (SIZES[0], 21) or covered.shape != (SIZES[0], 1)):
    raise ValueError("H=1 source feature dimensions differ")

text = np.load("fullpanel_H1_MiniLM_frozen.npy", mmap_mode="r")
centroids = np.load("fullpanel_H1_xgb_original_centroid_features.npy", mmap_mode="r")
centroid_ids = np.load("fullpanel_H1_centroids_original_record_ids.npy")
if (text.shape != (SIZES[0], 384) or centroids.shape != (SIZES[0], 1045)
        or not np.array_equal(centroid_ids, ids_h1)
        or not np.array_equal(centroids[:, :21], base)):
    raise ValueError("Semantic feature rows or dimensions differ")
source_h1 = {
    "paper": (np.column_stack((base, paper, covered)),
              names_base + names_paper + ["source_years_covered"],
              [("Early citations", 0, 21), ("Paper activity", 21, 63),
               ("Coverage", 63, 64)]),
    "finance": (np.column_stack((base, finance, covered)),
                names_base + names_finance + ["source_years_covered"],
                [("Early citations", 0, 21), ("Financial activity", 21, 42),
                 ("Coverage", 42, 43)]),
    "combined": (np.column_stack((base, paper, finance, covered)),
                 names_base + names_paper + names_finance + ["source_years_covered"],
                 [("Early citations", 0, 21), ("Paper activity", 21, 63),
                  ("Financial activity", 63, 84), ("Coverage", 84, 85)]),
}
selected = {k: json.load(open(p, encoding="utf-8")) for k, p in SELECTION.items()}
input_hash = {"citation": sha_file("fullpanel_citation_age0_9_H1_aligned.npz"),
              "feature": sha_file("fullpanel_scientific_market_features_H1.npz"),
              "text": sha_file("fullpanel_H1_MiniLM_frozen.npy"),
              "centroids": sha_file("fullpanel_H1_xgb_original_centroid_features.npy")}

def setting(family, H, ids_hash):
    root = selected[family]
    if family == "citations":
        e = root["selected_by_horizon"][str(H)]
        if e["data_hash"] != ids_hash: raise ValueError("Citation tuning IDs differ")
        return int(e["candidate"]), e["params"]
    if family == "text":
        if root["text_dimensions"] != 384: raise ValueError("Text dimension differs")
        return int(root["candidate"]), root["params"]
    if family == "centroids":
        e = root["horizons"][str(H)]
        if e["ids_hash"] != ids_hash: raise ValueError("Centroid tuning IDs differ")
        return int(e["candidate"]), e["parameters"]
    if (root["citation_sha256"] != input_hash["citation"]
            or root["feature_sha256"] != input_hash["feature"]):
        raise ValueError(f"{family}: tuned source files differ")
    e = root["settings_by_horizon"][str(H)]
    if e["ids_hash"] != ids_hash: raise ValueError(f"{family}: tuning IDs differ")
    return int(e["candidate"]), e["params"]

all_rows = []
for H, size in enumerate(SIZES, 1):
    mask = observed[:, :5 + H].all(axis=1)
    row_h1 = np.flatnonzero(mask)
    ids, years = ids_h1[mask], years_h1[mask]
    if len(ids) != size: raise ValueError(f"H={H}: eligible sample differs")
    ids_hash = hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest()
    y = history[mask, 5:5+H].sum(axis=1, dtype=np.int64)
    train, test = train_test_split(np.arange(size), test_size=.2, random_state=20)
    sample = np.sort(np.random.default_rng(2026+H).choice(len(test), 2000, replace=False))
    for family in ("citations", "text", "centroids", "paper", "finance", "combined"):
        candidate, params = setting(family, H, ids_hash)
        original, pred_path = saved_predictions(family, H, ids, years, y, test,
                                                candidate, params)
        tag = f"{family}_H{H}_seed20"
        stem = os.path.join(OUT, tag)
        model_path, shap_path = stem + "_model.ubj", stem + "_shap.npz"
        report_path = stem + "_comparison.json"
        if all(os.path.isfile(p) for p in (model_path, shap_path, report_path)):
            with open(report_path, encoding="utf-8") as f: result = json.load(f)
            if (result["saved_prediction_sha256"] != sha_file(pred_path)
                    or result["settings_sha256"] != sha_file(SELECTION[family])
                    or result["xgboost_version"] != xgboost.__version__
                    or result["input_hash"] != input_hash):
                raise ValueError(f"{tag}: saved refit provenance differs")
            all_rows.append(result)
            print(f"{tag}: prior refit verified; reused", flush=True)
            continue

        temp = None
        try:
            if family in source_h1:
                full, names, groups = source_h1[family]
                X = full[mask]
                xtrain, xtest = X[train], X[test]
            elif family == "citations":
                names, groups = names_base, [("Early citations", 0, 21)]
                xtrain, xtest = base[row_h1[train]], base[row_h1[test]]
            elif family == "text":
                names = names_base + [f"MiniLM_{j:03d}" for j in range(384)]
                groups = [("Early citations", 0, 21), ("Patent text", 21, 405)]
                xtrain = np.column_stack((base[row_h1[train]], text[row_h1[train]]))
                xtest = np.column_stack((base[row_h1[test]], text[row_h1[test]]))
            else:
                names = names_base + [f"Centroid_{j:04d}" for j in range(1024)]
                groups = [("Early citations", 0, 21), ("Centroids", 21, 1045)]
                fd, temp = tempfile.mkstemp(prefix="centroid_shap_", suffix=".npy", dir=OUT)
                os.close(fd)
                xtrain = np.lib.format.open_memmap(temp, mode="w+", dtype="float32",
                                                    shape=(len(train), 1045))
                for a in range(0, len(train), 4096):
                    b = min(a+4096, len(train))
                    xtrain[a:b] = centroids[row_h1[train[a:b]]]
                xtrain.flush()
                xtest = centroids[row_h1[test]].copy()
            sampled_X = np.asarray(xtest[sample]).copy()
            fit_y = np.log1p(y[train].astype(np.float32))
            model = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                 n_jobs=min(8, os.cpu_count() or 4),
                                 random_state=20 if family == "citations" else 42,
                                 **params)
            print(f"{tag}: fitting selected candidate {candidate}", flush=True)
            model.fit(xtrain, fit_y)
            del xtrain
            gc.collect()
            raw = model.predict(xtest)
            pred = np.clip(np.expm1(raw if family in SOURCE else raw.astype(np.float64)),
                           0, None)
            if family in SOURCE: pred = pred.astype(np.float32)
            del xtest
            gc.collect()
            diff = np.abs(pred - original)
            reproduced = bool(np.allclose(pred, original, atol=.001, rtol=.0001))
            prior_score, refit_score = score(y[test], original), score(y[test], pred)
            print(f"{tag}: match={reproduced}; mean difference={diff.mean():.6g}; "
                  f"saved/refit MAE={prior_score['MAE']:.4f}/{refit_score['MAE']:.4f}",
                  flush=True)
            booster = model.get_booster()
            contributions = booster.predict(DMatrix(sampled_X), pred_contribs=True)
            margins = booster.predict(DMatrix(sampled_X), output_margin=True)
            if (contributions.shape != (len(sample), len(names)+1)
                    or np.max(np.abs(contributions.sum(axis=1)-margins)) > 1e-4):
                raise ValueError(f"{tag}: SHAP additivity failed")
            values = contributions[:, :-1]
            tmp_model = stem + "_model.tmp.ubj"
            model.save_model(tmp_model)
            os.replace(tmp_model, model_path)
            tmp_shap = shap_path + ".tmp.npz"
            np.savez_compressed(tmp_shap, record_id=ids[test[sample]],
                                X_sample=sampled_X, shap_values=values,
                                base_values=contributions[:, -1],
                                feature_names=np.asarray(names, dtype="U64"))
            os.replace(tmp_shap, shap_path)
            pd.DataFrame({"feature": names,
                          "mean_abs_SHAP_log_scale": np.abs(values).mean(axis=0)}).sort_values(
                              "mean_abs_SHAP_log_scale", ascending=False).to_csv(
                                  stem + "_feature_importance.csv", index=False)
            pd.DataFrame([{"group": label,
                           "mean_abs_group_SHAP_log_scale": float(np.abs(values[:, a:b].sum(axis=1)).mean())}
                          for label, a, b in groups]).to_csv(stem + "_group_importance.csv",
                                                               index=False)
            result = {"family": family, "H": H, "seed": 20, "candidate": candidate,
                      "xgboost_version": xgboost.__version__,
                      "settings_sha256": sha_file(SELECTION[family]),
                      "saved_prediction_sha256": sha_file(pred_path),
                      "input_hash": input_hash,
                      "reproduced_within_tolerance": reproduced,
                      "explanation_scope": ("reproduces_saved_predictions_within_tolerance" if reproduced else
                                            "same_configuration_refit_only"),
                      "mean_abs_prediction_difference": float(diff.mean()),
                      "max_abs_prediction_difference": float(diff.max()),
                      **{f"saved_{k}": v for k, v in prior_score.items()},
                      **{f"refit_{k}": v for k, v in refit_score.items()}}
            tmp_json = report_path + ".tmp"
            with open(tmp_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            os.replace(tmp_json, report_path)
            all_rows.append(result)
            del model, booster, values, sampled_X, contributions, margins
        finally:
            gc.collect()
            if temp is not None and os.path.isfile(temp): os.remove(temp)
    pd.DataFrame(all_rows).to_csv(os.path.join(OUT, "comparison_summary.csv"), index=False)
    print(f"H={H}: six families complete", flush=True)

print(pd.DataFrame(all_rows)[["family", "H", "reproduced_within_tolerance",
                              "saved_MAE", "refit_MAE", "saved_p10", "refit_p10"]].to_string(index=False))
print("Saved models, SHAP values, importance tables, and checks in:", OUT)
