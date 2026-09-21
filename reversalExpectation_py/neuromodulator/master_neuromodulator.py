"""
master_neuromodulator.py
========================
Port of master_banditneuromodulator.m (H Atilgan & AC Kwan).

Full pipeline for the miniscope neuromodulator dataset:
    1. Scan log files and build data index
       1b. Fix 'animal' -- make_data_index() reads it from the parent folder
           name, which is wrong here because these logs sit flat in one
           folder instead of per-animal subfolders
       1c. Add 'date_number' -- needed by per_animal_neuromodulator's
           chronological sort, not computed by make_data_index()
    2. Determine each session's phase (needed to filter Phase==31 below --
       make_data_index() does not compute this; it only lists files)
    3. Add miniscope neural data metadata
    4. Create per-trial dF/F files (_dff.npz) for each session
    5. Per-session PSTH analysis
    6. Per-animal merged analysis
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from behavior.master_behavior import make_data_index
from preprocessing.log_parser import parse_logfile, detect_phase
from neuromodulator.add_index_neuromodulator import add_index_neuromodulator
from neuromodulator.create_dff_files         import create_dff_files
from neuromodulator.per_session              import per_session_neuromodulator
from neuromodulator.per_animal               import per_animal_neuromodulator


DATA_ROOT      = "data/data-behavior"
SUBFOLDER      = "bandit_R71_neuromodulator"
MINISCOPE_PATH = "data/data-miniscope"


def fix_animal_column(data_index: pd.DataFrame) -> pd.DataFrame:
    """
    make_data_index() sets 'animal' from the PARENT FOLDER name of each file
    (f.parent.name.lstrip("Mm")). That's correct when sessions live in
    per-animal subfolders, but the neuromodulator logs are flat in a single
    folder (bandit_R71_neuromodulator/), so every row ends up with
    animal == "bandit_R71_neuromodulator" -- breaks the miniscope CSV prefix
    AND would later lump every real animal into one fake group in
    per_animal_neuromodulator's groupby("animal").

    The real animal ID is the leading numeric token in session_file, e.g.
    "1807_phase3_..." -> "1807", "19303_phase31_..." -> "19303". Re-derive it
    from there instead.
    """
    animal = data_index["session_file"].str.extract(r"^(\d+)")[0]
    n_missing = animal.isna().sum()
    if n_missing:
        warnings.warn(f"[animal] {n_missing} session_file(s) didn't start with "
                       "digits -- couldn't recover an animal id, left as NaN.")
    out = data_index.copy()
    out["animal"] = animal
    print(f"[animal] recovered {out['animal'].nunique()} distinct animals "
          f"from session_file (was 1 fake animal from the parent folder name).")
    return out


def fix_date_number_column(data_index: pd.DataFrame) -> pd.DataFrame:
    """
    per_animal_neuromodulator() does group.sort_values("date_number") to put
    each animal's sessions in chronological order before concatenating them --
    but 'date_number' doesn't exist in data_index either (same gap as
    'animal' and 'phase': make_data_index() only lists files, it doesn't
    parse anything out of the names). Recover it from the trailing 10-digit
    timestamp in session_file, e.g. "..._1809161359.log" -> 1809161359.
    Without this, the per-animal step would crash the same way the phase
    filter did.
    """
    dn = data_index["session_file"].str.extract(r"_(\d{10})\.")[0]
    n_missing = dn.isna().sum()
    if n_missing:
        warnings.warn(f"[date_number] {n_missing} session_file(s) had no "
                       "10-digit timestamp -- left as NaN.")
    out = data_index.copy()
    out["date_number"] = pd.to_numeric(dn)
    return out


def add_phase_column(data_index: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'phase' column to data_index by reading each session's scenario
    name -- the piece of information make_data_index() doesn't compute
    (it only lists files, it doesn't parse them). .mat rows get their phase
    from the saved trialData.presCodeSet; .log rows get it by reading the
    log header via parse_logfile() + detect_phase() (cheap -- header only,
    no full trial parsing). Rows that fail to parse get phase = NaN and are
    reported, not silently dropped.
    """
    phases = []
    n_failed = 0
    for _, row in data_index.iterrows():
        try:
            if row["file_type"] == "log":
                log_data = parse_logfile(row["beh_path"])
                phases.append(detect_phase(log_data["scenario"]))
            else:  # "mat"
                from scipy.io import loadmat
                mat = loadmat(row["beh_path"], squeeze_me=True, struct_as_record=False)
                phases.append(int(mat["trialData"].presCodeSet))
        except Exception as e:
            phases.append(np.nan)
            n_failed += 1
            warnings.warn(f"Could not determine phase for {row['session_file']}: {e}")

    out = data_index.copy()
    out["phase"] = phases
    if n_failed:
        print(f"[phase] {n_failed} / {len(out)} sessions had no readable phase "
              "(kept as NaN, excluded by the ==31 filters below).")
    print(f"[phase] value counts:\n{out['phase'].value_counts(dropna=False).to_string()}")
    return out


def build_index(
    data_root: str = DATA_ROOT,
    subfolder: str = SUBFOLDER,
    miniscope_path: str = MINISCOPE_PATH,
) -> pd.DataFrame:
    """Pipeline steps 1-3 (index + animal + date_number + phase + miniscope).
    Single source of truth: used by run(), the inventory and the QC scripts."""
    data_index = make_data_index(data_root, subfolder)
    if data_index.empty:
        return data_index
    data_index = fix_animal_column(data_index)
    data_index = fix_date_number_column(data_index)
    data_index = add_phase_column(data_index)
    return add_index_neuromodulator(data_index, miniscope_path)


def run(
    data_root: str = DATA_ROOT,
    subfolder: str = SUBFOLDER,
    miniscope_path: str = MINISCOPE_PATH,
):
    # 1-3. Index + animal + date_number + phase + miniscope data
    data_index = build_index(data_root, subfolder, miniscope_path)

    if data_index.empty:
        print("No neuromodulator sessions found — skipping.")
        return [], []
    print(f"\nSessions with cell CSV:       {data_index['cell_created'].notna().sum()}")
    print(f"Sessions with timeEvents CSV: {data_index['time_events_created'].notna().sum()}")

    # 4. Create per-trial dF/F files
    data_index = create_dff_files(data_index)
    print(f"Sessions with dff file:       {data_index['dff_created'].notna().sum()}")

    figurepath = Path(data_root).parent / "figs"

    # 5. Per-session analysis (phase 31, experiment 1 = NE signal)
    #    Matches master_banditneuromodulator.m:
    #    criteriaSubset = (ismember(dataIndex.Phase,31)==1) & dataIndex.experiment==1
    #                       & ~isnan(dataIndex.dffCreated);
    ne_sessions = data_index[
        (data_index["phase"] == 31) &
        (data_index["experiment"] == 1) &
        data_index["dff_created"].notna()
    ]
    print(f"\nRunning per-session analysis on {len(ne_sessions)} NE sessions...")
    session_results = per_session_neuromodulator(
        ne_sessions,
        save_path=str(figurepath / "neuromodulator-per-session-ne"),
    )
    print(f"Done — {len(session_results)} sessions analysed.")

    # 6. Per-animal analysis (all Phase-31 sessions with dff — no experiment
    #    filter here, matching the .m original):
    #    criteriaSubset = (ismember(dataIndex.Phase,31)==1) & ~isnan(dataIndex.dffCreated);
    all_sessions = data_index[
        (data_index["phase"] == 31) &
        data_index["dff_created"].notna()
    ]
    print(f"\nRunning per-animal analysis on {len(all_sessions)} sessions...")
    animal_results = per_animal_neuromodulator(
        all_sessions,
        save_path=str(figurepath / "neuromodulator-per-animal"),
    )
    print(f"Done — {len(animal_results)} animals analysed.")
    return session_results, animal_results


if __name__ == "__main__":
    run()
