from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd

# Put these four files in the notebook's working directory.
root = Path.cwd()
files = {
    "panel": root / "citation_panel_han.csv",
    "papers": root / "paper_cluster_match_v2_filtered.csv",
    "paper_year": root / "paper_cluster_year_frequency_v2_filtered(1).csv",
    "finance_year": root / "financial_cluster_year_frequency_v2_filtered(1).csv",
}
for name, path in files.items():
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")

paper = pd.read_csv(files["papers"], usecols=[
    "UT (Unique WOS ID)", "Publication Year", "hit_cluster_ids",
    "n_clusters_hit", "hit_cluster_freq_json",
])
paper_y = pd.read_csv(files["paper_year"])
fin_y = pd.read_csv(files["finance_year"])

for name, table in [("paper", paper_y), ("finance", fin_y)]:
    assert list(table.columns) == ["Cluster_ID", "Year", "Hit_Count"]
    assert not table.isna().any().any()
    assert not table.duplicated(["Cluster_ID", "Year"]).any()
    assert (table["Hit_Count"] > 0).all()
    print(f"{name}: {len(table):,} cluster-year rows; "
          f"{table.Cluster_ID.nunique():,} IDs; "
          f"years {table.Year.min()}–{table.Year.max()}; "
          f"total hits {table.Hit_Count.sum():,}")

assert not paper["UT (Unique WOS ID)"].duplicated().any()
assigned = paper.hit_cluster_ids.fillna("").map(
    lambda raw: len({x for x in str(raw).split(";") if x.strip()})
)
assert assigned.eq(paper.n_clusters_hit).all()

# Determine precisely what the supplied paper Hit_Count means.
recount = Counter()
distinct_papers = Counter()
for year, ids_raw, freq_raw in zip(
    paper["Publication Year"], paper.hit_cluster_ids,
    paper.hit_cluster_freq_json,
):
    y = int(year)
    if y > 2025:
        continue
    for token in set(str(ids_raw).split(";")) - {"", "nan"}:
        distinct_papers[(int(token), y)] += 1
    if pd.notna(freq_raw):
        for cid, hits in json.loads(freq_raw).items():
            recount[(int(cid), y)] += int(hits)

supplied = {
    (int(cid), int(year)): int(n)
    for cid, year, n in paper_y.itertuples(index=False, name=None)
}
differences = [
    key for key in recount.keys() | supplied.keys()
    if recount.get(key, 0) != supplied.get(key, 0)
]
print(f"paper records: {len(paper):,}; publication year range "
      f"{paper['Publication Year'].min()}–{paper['Publication Year'].max()}")
print(f"paper annual table vs JSON keyword-hit recount: "
      f"{len(differences):,} differing cluster-years")
print(f"distinct-paper assignments through 2025: "
      f"{sum(distinct_papers.values()):,}; "
      f"keyword hits through 2025: {sum(recount.values()):,}")
if differences:
    print("Example discrepancies:", [
        (key, recount.get(key, 0), supplied.get(key, 0))
        for key in sorted(differences)[:5]
    ])

# Scan the full panel in chunks. This does not build feature matrices or fit a model.
max_age_parts = []
origin_parts = []
for chunk in pd.read_csv(
    files["panel"],
    usecols=["record_id", "pub_year", "age", "clusters"],
    chunksize=250_000,
):
    max_age_parts.append(chunk.groupby("record_id")["age"].max())
    origin_parts.append(chunk.loc[
        chunk.age.eq(0), ["record_id", "pub_year", "clusters"]
    ])

max_age = pd.concat(max_age_parts).groupby(level=0).max().sort_index()
origin = pd.concat(origin_parts, ignore_index=True)
assert not origin.record_id.duplicated().any()
origin = origin.set_index("record_id").sort_index()
assert origin.index.equals(max_age.index)

expected_counts = {1: 268192, 2: 248874, 3: 225128,
                   4: 201400, 5: 178765}
expected_hashes = {
    1: "d0942b5bd4cf", 2: "6a8ee6fd7cab", 3: "a1065924d755",
    4: "72c6ac5f90a2", 5: "2a3a91b3a7f8",
}
horizon_ids = {}
for H in range(1, 6):
    ids = max_age.index[max_age.ge(4 + H)].to_numpy(np.int64)
    fp = hashlib.sha256(ids.tobytes()).hexdigest()
    assert len(ids) == expected_counts[H] and fp.startswith(expected_hashes[H])
    horizon_ids[H] = ids

h1_ids = horizon_ids[1]
origin_h1 = origin.loc[h1_ids]
years = origin_h1.pub_year.to_numpy(dtype=np.int32)
cluster_lists = [
    np.fromiter(
        {int(t) for t in str(raw).split("|") if t.strip()},
        dtype=np.int32,
    )
    for raw in origin_h1.clusters.fillna("")
]

paper_ids = set(paper_y.Cluster_ID)
finance_ids = set(fin_y.Cluster_ID)
paper_id_present = np.fromiter(
    (any(cid in paper_ids for cid in cs) for cs in cluster_lists),
    dtype=bool, count=len(h1_ids),
)
fin_id_present = np.fromiter(
    (any(cid in finance_ids for cid in cs) for cs in cluster_lists),
    dtype=bool, count=len(h1_ids),
)

# Age-0 through age-4 calendar years only. No future paper or finance years.
year_start = min(int(paper_y.Year.min()), int(fin_y.Year.min()))
year_end = max(int(paper_y.Year.max()), int(fin_y.Year.max()))
n_clust = max(
    int(paper_y.Cluster_ID.max()), int(fin_y.Cluster_ID.max()),
    max(int(cs.max()) for cs in cluster_lists if len(cs)),
) + 1

def prefix_table(table):
    matrix = np.zeros((n_clust, year_end - year_start + 1), dtype=np.int64)
    matrix[
        table.Cluster_ID.to_numpy(),
        table.Year.to_numpy() - year_start,
    ] = table.Hit_Count.to_numpy()
    return np.pad(np.cumsum(matrix, axis=1), ((0, 0), (1, 0)))

paper_prefix = prefix_table(paper_y)
finance_prefix = prefix_table(fin_y)
paper_hits = np.zeros(len(h1_ids), dtype=np.int64)
finance_hits = np.zeros(len(h1_ids), dtype=np.int64)
incomplete_start = np.zeros(len(h1_ids), dtype=bool)

for i, (pub_year, cs) in enumerate(zip(years, cluster_lists)):
    cutoff = int(pub_year) + 4
    start_year = int(pub_year)
    incomplete_start[i] = start_year < year_start
    if len(cs) == 0 or cutoff < year_start or cutoff > year_end:
        continue
    start_col = max(start_year, year_start) - year_start
    end_col = cutoff - year_start + 1
    paper_hits[i] = (
        paper_prefix[cs, end_col] - paper_prefix[cs, start_col]
    ).sum()
    finance_hits[i] = (
        finance_prefix[cs, end_col] - finance_prefix[cs, start_col]
    ).sum()

index = pd.Index(h1_ids)
for H, ids in horizon_ids.items():
    positions = index.get_indexer(ids)
    assert (positions >= 0).all()
    print(
        f"H={H}: {len(ids):,} patents | "
        f"paper ID present {paper_id_present[positions].sum():,} | "
        f"finance ID present {fin_id_present[positions].sum():,} | "
        f"both IDs present {(paper_id_present[positions] & fin_id_present[positions]).sum():,} | "
        f"paper age0–4 hits >0 {(paper_hits[positions] > 0).sum():,} | "
        f"finance age0–4 hits >0 {(finance_hits[positions] > 0).sum():,} | "
        f"source starts after patent publication {incomplete_start[positions].sum():,}"
    )

for name, values in [("paper keyword hits", paper_hits),
                     ("financial hits", finance_hits)]:
    print(f"H=1 {name} age0–4: "
          f"median={np.median(values):.0f}, "
          f"p90={np.quantile(values, .90):.0f}, "
          f"max={values.max():,}")

print("Audit only: no modeling features saved and no model fitted.")
