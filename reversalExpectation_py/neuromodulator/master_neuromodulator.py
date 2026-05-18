"""
master_neuromodulator.py
========================
Port of master_banditneuromodulator.m (H Atilgan & AC Kwan).

Full pipeline for the miniscope neuromodulator dataset:
    1. Scan log files and build data index
    2. Add miniscope neural data metadata
    3. Create per-trial dF/F files (_dff.npz) for each session
    4. Per-session PSTH analysis
    5. Per-animal merged analysis
"""

import sys
from pathlib import Path


# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from behavior.master_behavior import make_data_index
from neuromodulator.add_index_neuromodulator import add_index_neuromodulator
from neuromodulator.create_dff_files         import create_dff_files
from neuromodulator.per_session              import per_session_neuromodulator
from neuromodulator.per_animal               import per_animal_neuromodulator


DATA_ROOT      = "data/data-behavior"
SUBFOLDER      = "bandit_neuromodulator/data"
MINISCOPE_PATH = "data/data-miniscope"


def run(
    data_root: str = DATA_ROOT,
    subfolder: str = SUBFOLDER,
    miniscope_path: str = MINISCOPE_PATH,
):
    # 1. Build behavioral data index
    data_index = make_data_index(data_root, subfolder)

    if data_index.empty:
        print("No neuromodulator sessions found — skipping.")
        return [], []

    # 2. Add miniscope neural data metadata
    data_index = add_index_neuromodulator(data_index, miniscope_path)
    print(f"\nSessions with cell CSV:       {data_index['cell_created'].notna().sum()}")
    print(f"Sessions with timeEvents CSV: {data_index['time_events_created'].notna().sum()}")

    # 3. Create per-trial dF/F files
    data_index = create_dff_files(data_index)
    print(f"Sessions with dff file:       {data_index['dff_created'].notna().sum()}")

    figurepath = Path(data_root).parent / "figs"

    # 4. Per-session analysis (phase 31, experiment 1 = NE signal)
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

    # 5. Per-animal analysis (all sessions with dff)
    all_sessions = data_index[data_index["dff_created"].notna()]
    print(f"\nRunning per-animal analysis on {len(all_sessions)} sessions...")
    animal_results = per_animal_neuromodulator(
        all_sessions,
        save_path=str(figurepath / "neuromodulator-per-animal"),
    )
    print(f"Done — {len(animal_results)} animals analysed.")
    return session_results, animal_results


if __name__ == "__main__":
    run()
