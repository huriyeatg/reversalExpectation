"""
add_index_neuromodulator.py
===========================
Port of addIndexNeuromodulator.m (H Atilgan & AC Kwan, 20092020).

Scans the miniscope data folder for CSV files matching each session and
appends neural data metadata columns to the data index DataFrame.
"""

import math
from pathlib import Path

import pandas as pd


def add_index_neuromodulator(data_index: pd.DataFrame, signal_path: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    data_index  : DataFrame from make_data_index (master_bandit.py)
    signal_path : path to the miniscope data folder (data/data-miniscope)

    Returns
    -------
    DataFrame with added columns:
        experiment           : 1 = norepinephrine, 2 = acetylcholine
        cell_created         : 1.0 if *_cell.csv found, else NaN
        time_events_created  : 1.0 if *_timeEvents.csv found, else NaN
        cell_filename        : matched cell CSV filename
        time_events_filename : matched timeEvents CSV filename
        neural_data_path     : path to signal_path
    """
    signal_path = Path(signal_path)

    records = []
    for _, row in data_index.iterrows():
        animal = str(row["animal"])
        phase  = int(row.get("phase", 31))
        dn     = row.get("date_number", float("nan"))
        blockdate = str(int(dn))[:6] if not (isinstance(dn, float) and math.isnan(dn)) else ""

        # Experiment type: 3rd character of animal ID → 1=NE, 2=ACh
        experiment = float("nan")
        if len(animal) >= 3:
            if animal[2] == "3":
                experiment = 1.0
            elif animal[2] == "4":
                experiment = 2.0

        prefix = f"M{animal}_Phase{phase}_{blockdate}"

        cell_files = sorted(signal_path.glob(f"{prefix}*_cell.csv"))
        cell_created  = 1.0 if cell_files else float("nan")
        cell_filename = cell_files[0].name if cell_files else ""

        te_files = sorted(signal_path.glob(f"{prefix}*_timeEvents.csv"))
        te_created  = 1.0 if te_files else float("nan")
        te_filename = te_files[0].name if te_files else ""

        records.append({
            "experiment":           experiment,
            "cell_created":         cell_created,
            "time_events_created":  te_created,
            "cell_filename":        cell_filename,
            "time_events_filename": te_filename,
            "neural_data_path":     str(signal_path),
        })

    _cols = [
        "experiment", "cell_created", "time_events_created",
        "cell_filename", "time_events_filename", "neural_data_path",
    ]
    if records:
        nm_df = pd.DataFrame(records, index=data_index.index)
    else:
        nm_df = pd.DataFrame(columns=_cols)
    return pd.concat([data_index, nm_df], axis=1)
