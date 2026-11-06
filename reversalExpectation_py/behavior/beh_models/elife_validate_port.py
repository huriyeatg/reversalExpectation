"""
elife_validate_port.py
======================
Validates the Python port of the eLife models (elife_models.py) against the
MATLAB fits in sim/mat_models/ (bandit_fittingPerAnimal.m, 31 naive mice).

1. Recover the exact sessions MATLAB used. sim/mat_stats/experimental_stats.mat
   holds the concatenated choice/outcome sequence that was fitted (all animals,
   sessions joined by merge_sessions.m with 20 NaN trials). Every session of
   the behavioural CSV is located in that sequence by exact match of its
   choice/outcome string; sessions found are the MATLAB fitting set, in the
   MATLAB order.
2. For every animal and model, evaluate the negative log-likelihood of the port
   at the MATLAB fitted parameters on the same concatenated sequence, and
   compare it with the stored negloglike. A port is exact when they agree to
   numerical precision.
3. The same evaluation with the exact Bayes update (literal=False) shows how
   much the MATLAB right-choice update of funbelief/funbelief_CK matters.

Inputs:
    reversalExpectation_py/analysis/bandit_R71_lesion.csv
    sim/mat_stats/experimental_stats.mat
    sim/mat_models/<model>.mat
Outputs (reversalExpectation_py/analysis/elife_port_validation/):
    matlab_sessions.csv       sessions used by the MATLAB per-animal fits
    port_vs_matlab.csv        per animal x model: MATLAB vs port negloglike
    report.txt

Usage (from reversalExpectation_py/):
    python behavior/beh_models/elife_validate_port.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import elife_models as em                    # noqa: E402

PY_ROOT = HERE.parents[1]                    # reversalExpectation_py/
REPO = PY_ROOT.parent
CSV_PATH = PY_ROOT / "analysis" / "bandit_R71_lesion.csv"
SIM_DIR = REPO / "sim"
OUT_DIR = PY_ROOT / "analysis" / "elife_port_validation"
MODELS_TO_CHECK = em.ELIFE8 + ["belief_CK_bilateral"]


def _encode(c, r):
    """One character per trial: a/b = left rewarded/unrewarded, c/d = right, N = miss."""
    c, r = np.asarray(c, float), np.asarray(r, float)
    out = np.full(len(c), "N", dtype="<U1")
    out[(c == -1) & (r == 1)] = "a"
    out[(c == -1) & (r == 0)] = "b"
    out[(c == 1) & (r == 1)] = "c"
    out[(c == 1) & (r == 0)] = "d"
    return "".join(out)


def matlab_sessions(df: pd.DataFrame, stats_mat: Path) -> pd.DataFrame:
    """Locate every CSV session in the MATLAB fitted sequence."""
    st = loadmat(stats_mat, squeeze_me=True, struct_as_record=False)["stats"]
    big = _encode(st.c, st.r)
    rows = []
    for (a, s), d in df.groupby(["animal", "session_file"], sort=False):
        d = d.sort_values("trial_idx")
        code = _encode(d["choice"], d["rewarded"])
        pos = big.find(code)
        if pos >= 0:
            rows.append({"animal": a, "session_file": s, "pos": pos, "n_trials": len(d),
                         "n_responded": int(d["choice"].notna().sum()),
                         "meets_criteria": bool(d["meets_criteria"].iloc[0]),
                         "lesioned": d["lesioned"].iloc[0]})
    t = pd.DataFrame(rows).sort_values("pos").reset_index(drop=True)
    # a short session can match inside a longer one: drop matches contained in
    # another match. (No filter on the CSV `lesioned` flag: one 1808 session is
    # flagged lesioned in the CSV but was fitted by MATLAB as a naive session.)
    end = t["pos"] + t["n_trials"]
    contained = np.zeros(len(t), bool)
    for i in range(len(t)):
        others = (t["pos"] <= t["pos"][i]) & (end >= end[i]) & (t.index != i)
        contained[i] = bool(others.any())
    t = t[~contained].reset_index(drop=True)
    return t, big


def animal_sequence(df, ses_table, animal):
    """(c, r, starts) of one animal in the MATLAB layout, plus the session list."""
    ses = ses_table[ses_table["animal"] == animal].sort_values("pos")
    parts = []
    for s in ses["session_file"]:
        d = df[(df["animal"] == animal) & (df["session_file"] == s)].sort_values("trial_idx")
        parts.append((d["choice"].to_numpy(float), d["rewarded"].to_numpy(float)))
    return em.build_concat(parts), ses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--sim", type=Path, default=SIM_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)
    df["animal"] = df["animal"].astype(str)
    ses, big = matlab_sessions(df, args.sim / "mat_stats" / "experimental_stats.mat")
    ses.to_csv(args.out / "matlab_sessions.csv", index=False)

    # check: our reconstruction of each animal's block equals the MATLAB sequence
    seq_ok = {}
    for a in ses["animal"].unique():
        (c, r, _), s = animal_sequence(df, ses, a)
        p0 = int(s["pos"].min())
        seq_ok[a] = big[p0:p0 + len(c)] == _encode(c, r)

    rows = []
    for name in MODELS_TO_CHECK:
        f = args.sim / "mat_models" / f"{name}.mat"
        if not f.exists():
            continue
        m = loadmat(f, squeeze_me=True, struct_as_record=False)
        for a, par, nll in zip(m["animal"], m["fitpar"], m["negloglike"]):
            a = str(a)
            if a not in seq_ok:
                rows.append({"model": name, "animal": a, "status": "no_sessions"})
                continue
            (c, r, starts), _ = animal_sequence(df, ses, a)
            par = np.atleast_1d(np.asarray(par, float))
            port = em.negloglike(name, par, c, r, literal=True)
            exact = em.negloglike(name, par, c, r, literal=False)
            rows.append({"model": name, "animal": a, "status": "ok",
                         "sequence_identical": bool(seq_ok[a]),
                         "nll_matlab": float(nll), "nll_port": port,
                         "abs_diff": abs(port - float(nll)),
                         "nll_exact_bayes": exact,
                         "n_scored": em.n_scored(name, par, c, r),
                         **{f"par_{lbl}": v for lbl, v in zip(em.MODELS[name].labels, par)}})
    res = pd.DataFrame(rows)
    res.to_csv(args.out / "port_vs_matlab.csv", index=False)

    ok = res[res["status"] == "ok"]
    summ = (ok.groupby("model")
              .agg(n_animals=("animal", "size"),
                   max_abs_diff=("abs_diff", "max"),
                   median_abs_diff=("abs_diff", "median"),
                   total_nll_matlab=("nll_matlab", "sum"),
                   total_nll_port=("nll_port", "sum"),
                   total_nll_exact_bayes=("nll_exact_bayes", "sum"))
              .reindex([m for m in MODELS_TO_CHECK if m in set(ok["model"])]))
    lines = [
        "eLife model port vs MATLAB (bandit_fittingPerAnimal.m, sim/mat_models)",
        "=" * 72,
        f"MATLAB fitting set recovered: {len(ses)} sessions, {ses['animal'].nunique()} animals, "
        f"{int(ses['n_responded'].sum())} responded trials "
        f"({int(ses['meets_criteria'].sum())} meet the current criteria, "
        f"{int((~ses['meets_criteria']).sum())} do not)",
        f"animal sequences identical to experimental_stats.mat: "
        f"{sum(seq_ok.values())}/{len(seq_ok)}",
        "",
        "negloglike at the MATLAB parameters (port with literal=True must match):",
        summ.to_string(float_format=lambda v: f"{v:.6g}"),
    ]
    (args.out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
