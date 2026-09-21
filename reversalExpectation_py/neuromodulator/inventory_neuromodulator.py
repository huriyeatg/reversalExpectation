"""
inventory_neuromodulator.py
===========================
Step 0 of the neuromodulator analysis: inventory + quality control.

The index is built with master_neuromodulator.build_index(), the single
source of truth shared with the pipeline (paths, animal / date_number /
phase fixes, miniscope matching).

Questions answered:
  (A) Design: animals x phase x experiment (NE/ACh in the same animals?)
  (B) Quality of the existing _dff.npz files (extreme values, NaN, short sessions)
  (C) Raw-trace format: number of columns/ROIs and whether it is positive raw F
  (D) GLM-HMM coverage for these animals

Run from reversalExpectation_py:
    python neuromodulator\\inventory_neuromodulator.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- Everything that defines the index comes from the master (same config as the pipeline) ---
import neuromodulator.master_neuromodulator as master

DATA_ROOT      = master.DATA_ROOT
SUBFOLDER      = master.SUBFOLDER
MINISCOPE_PATH = master.MINISCOPE_PATH

# GLM-HMM K=2 states file (adjust the path if needed; if missing, (D) is skipped)
GLMHMM_STATES = "data/glmhmm_states_K2.csv"
OUT_DIR = Path(DATA_ROOT).parent / "figs" / "neuromodulator-qc"

# QC thresholds for the existing _dff.npz files
MAX_ABS_DFF, MAX_NAN_FRAC, MIN_TRIALS = 5.0, 0.5, 100


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------
def where_are_the_logs(root: str) -> None:
    """Diagnostic when the index is empty: counts .log files per folder under root."""
    rootp = Path(root)
    print(f"\n[diagnostic] DATA_ROOT = {rootp.resolve()}  (exists: {rootp.exists()})")
    print(f"[diagnostic] master SUBFOLDER = '{SUBFOLDER}'")
    if not rootp.exists():
        return
    counts = pd.Series([str(p.parent.relative_to(rootp)) for p in rootp.rglob("*.log")])
    if counts.empty:
        print("[diagnostic] No .log files under DATA_ROOT.")
    else:
        print("[diagnostic] Folders with .log files (use one of these as SUBFOLDER):")
        print(counts.value_counts().to_string())


def build_index() -> pd.DataFrame:
    """Same index as the pipeline: delegates to master_neuromodulator.build_index()."""
    return master.build_index()


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------
def qc_dff_npz(path: Path) -> dict:
    """Metrics of the existing _dff.npz (old normalization)."""
    dff = np.load(path)["dff"]
    vals = dff[np.isfinite(dff)]
    if not vals.size:
        return {"n_trials_dff": dff.shape[0], "nan_frac": 1.0}
    return {
        "n_trials_dff": dff.shape[0],
        "nan_frac": 1 - np.isfinite(dff).mean(),
        "dff_min": vals.min(), "dff_max": vals.max(), "dff_median": np.median(vals),
        "n_trials_extreme": int((np.nanmax(np.abs(dff), axis=1) > MAX_ABS_DFF).sum()),
    }


def qc_raw_trace(row) -> dict:
    """Reads the trace CSV like create_dff_files.py (skiprows=2) and reports
    how many signal columns there are and whether they look like raw fluorescence."""
    try:
        path = Path(row["neural_data_path"]) / row["cell_filename"]
        raw = pd.read_csv(path, skiprows=2, header=None)
        sig = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")   # col 0 = time
        # the 2 header rows usually hold each ROI's name and status
        header = pd.read_csv(path, nrows=2, header=None).iloc[:, 1:]
        col1 = sig.iloc[:, 0].to_numpy()
        return {
            "n_trace_cols": sig.shape[1],
            "header_row1": "|".join(map(str, header.iloc[0].tolist()[:5])),
            "header_row2": "|".join(map(str, header.iloc[1].tolist()[:5])) if len(header) > 1 else "",
            "raw_min": np.nanmin(col1), "raw_median": np.nanmedian(col1),
            "raw_frac_nonpos": float(np.mean(col1 <= 0)),
            # raw F: strictly positive. Otherwise dividing by F0 ~ 0 blows up.
            "looks_raw_F": bool(np.nanpercentile(col1, 0.1) > 0),
        }
    except Exception as e:
        return {"raw_error": repr(e)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = build_index()

    if idx.empty or "session_file" not in idx.columns:
        print("\nThe index is empty: no sessions found with this configuration.")
        where_are_the_logs(DATA_ROOT)
        sys.exit(1)

    idx["session_stem"] = idx["session_file"].map(lambda s: Path(str(s)).stem)
    print(f"\nSessions in index: {len(idx)} | columns: {list(idx.columns)}")

    # ---------------- (A) Design ----------------
    keys = [k for k in ("animal", "phase", "experiment") if k in idx.columns]
    design = idx.groupby(keys, dropna=False).size().rename("n_sessions").reset_index()
    design.to_csv(OUT_DIR / "design_table.csv", index=False)
    print("\n=== (A) Sessions per animal x phase x experiment ===")
    if {"phase", "experiment"} <= set(keys):
        print(design.pivot_table(index="animal", columns=["phase", "experiment"],
                                 values="n_sessions", fill_value=0).to_string())
        n_exp = idx.groupby("animal")["experiment"].nunique()
        print(f"\nAnimals with >1 'experiment': {int((n_exp > 1).sum())}/{len(n_exp)}")
    else:
        print(design.to_string(index=False))
        print(f"(columns missing for the pivot: {set(['phase', 'experiment']) - set(keys)})")
    # The protocol in the file name also helps (...WithPupilNE vs ...NM)
    idx["protocol"] = idx["session_stem"].str.split("_").str[2]
    by = ["protocol"] + (["experiment"] if "experiment" in idx.columns else [])
    print("\nProtocols in file names:")
    print(idx.groupby(by, dropna=False).size().to_string())

    # ---------------- (B) + (C) QC per session ----------------
    dff_files = {p.name: p for p in Path(DATA_ROOT).rglob("*_dff.npz")}
    rows = []
    for _, r in idx.iterrows():
        base = {k: r.get(k) for k in keys}
        base["session"] = r["session_stem"]
        f = dff_files.get(f"{r['session_stem']}_dff.npz")
        if f is not None:
            try:
                base.update(qc_dff_npz(f))
            except Exception as e:
                base["dff_error"] = repr(e)
        if pd.notna(r.get("cell_filename")) and pd.notna(r.get("neural_data_path")):
            base.update(qc_raw_trace(r))
        rows.append(base)
    qc = pd.DataFrame(rows)
    qc.to_csv(OUT_DIR / "session_qc.csv", index=False)

    print("\n=== (B) Existing _dff.npz ===")
    if "dff_max" in qc.columns:
        ext = qc[(qc["dff_max"].abs() > MAX_ABS_DFF) | (qc["dff_min"].abs() > MAX_ABS_DFF)]
        print(f"With _dff.npz: {qc['n_trials_dff'].notna().sum()} | "
              f"with extreme values: {len(ext)} | "
              f"NaN>{MAX_NAN_FRAC:.0%}: {int((qc['nan_frac'] > MAX_NAN_FRAC).sum())} | "
              f"<{MIN_TRIALS} trials: {int((qc['n_trials_dff'] < MIN_TRIALS).sum())}")
        if len(ext):
            print(ext[["session", "dff_min", "dff_max", "dff_median", "n_trials_extreme"]]
                  .sort_values("dff_max", key=np.abs, ascending=False)
                  .to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    else:
        print("No _dff.npz found under DATA_ROOT.")

    print("\n=== (C) Raw trace in the miniscope CSV ===")
    if "n_trace_cols" in qc.columns:
        print("Signal columns per session (value counts):")
        print(qc["n_trace_cols"].value_counts(dropna=False).to_string())
        print(f"Sessions that do NOT look like positive raw F: "
              f"{int((qc['looks_raw_F'] == False).sum())}/{qc['looks_raw_F'].notna().sum()}")
        print("\nExample headers (first sessions):")
        print(qc[["session", "n_trace_cols", "header_row1", "header_row2",
                  "raw_min", "raw_median", "looks_raw_F"]].head(5).to_string(index=False))
    if "raw_error" in qc.columns and qc["raw_error"].notna().any():
        print(f"Errors reading CSV: {qc['raw_error'].notna().sum()} "
              f"(e.g.: {qc['raw_error'].dropna().iloc[0]})")

    # ---------------- (D) GLM-HMM coverage ----------------
    print("\n=== (D) GLM-HMM state coverage ===")
    gp = Path(GLMHMM_STATES)
    if not gp.exists():
        print(f"{gp} not found — set GLMHMM_STATES.")
    else:
        st = pd.read_csv(gp, usecols=["animal", "session_file"])
        inter = sorted(set(idx["animal"].astype(str)) & set(st["animal"].astype(str)))
        st_stems = set(st["session_file"].map(lambda s: Path(str(s)).stem))
        n_sess = idx["session_stem"].isin(st_stems).sum()
        print(f"Imaging animals present in the states file: {len(inter)} {inter}")
        print(f"Sessions with states already inferred: {n_sess}/{len(idx)}")

    print(f"\nCSVs saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
