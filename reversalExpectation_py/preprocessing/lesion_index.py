"""
lesion_index.py
===============
Translations of addIndexLesion.m + determineBehCriteria.m
(H Atilgan & AC Kwan, 191127 / 191203).

LESION_SIDE: 1=Left, 2=Right, 3=Bilateral, 4=Saline, NaN=no lesion
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Lesion record (hardcoded in MATLAB, replicated exactly)
# ---------------------------------------------------------------------------
# Format: animal_id → (surgery_date_int, lesion_side)
# Date format: YYMMDDHHMM  e.g. 1806290000 = 2018 Jun 29 00:00

LESION_RECORD = {
    "1806":  (1806290000, 1),   # Left
    "1807":  (1806290000, 2),   # Right
    "1808":  (1810030000, 1),   # Left
    "19118": (1907220000, 4),   # Saline
    "18102": (1902240000, 1),   # Left
    "18103": (1902240000, 4),   # Saline
    "18104": (1902240000, 2),   # Right
    "18106": (1902140000, 3),   # Bilateral
    "18107": (1902140000, 1),   # Left
    "18109": (1902140000, 2),   # Right
    "19102": (1905200000, 3),   # Bilateral
    "19106": (1905200000, 3),   # Bilateral
    "19107": (1905200000, 1),   # Left
    "19109": (1905200000, 2),   # Right
    "19114": (1908260000, 4),   # Saline
    "19116": (1907220000, 4),   # Saline
    "19117": (1907220000, 3),   # Bilateral
}

LESION_SIDE_LABELS = {1: "Left", 2: "Right", 3: "Bilateral", 4: "Saline"}


def add_lesion_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translation of addIndexLesion.m.

    Adds 'lesioned' and 'lesion_side' columns to the session DataFrame.

    Parameters
    ----------
    df : DataFrame with columns 'animal' and 'date_number'

    Returns
    -------
    df with new columns:
        lesioned     : NaN = control/pre-lesion, 1 = post-lesion
        lesion_side  : NaN = no lesion, 1=Left, 2=Right, 3=Bilateral, 4=Saline
    """
    df = df.copy()
    df["lesioned"]    = np.nan
    df["lesion_side"] = np.nan

    for i, row in df.iterrows():
        animal = str(row["animal"])
        if animal in LESION_RECORD:
            surgery_date, side = LESION_RECORD[animal]
            df.at[i, "lesion_side"] = float(side)
            if row["date_number"] > surgery_date:
                df.at[i, "lesioned"] = 1.0

    return df


# ---------------------------------------------------------------------------
# determineBehCriteria.m
# ---------------------------------------------------------------------------

NUM_TRIAL_CRITERION  = 100   # min responsive trials
NUM_SWITCH_CRITERION = 3     # min rule switches: keep sessions with switch_num > 3,
                             # i.e. >= 4 block switches. Matches Murphy et al. 2024
                             # ("excluded the session if the animal had fewer than 4
                             # block switches", Methods)


def compute_session_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translation of determineBehCriteria.m.

    Uses precomputed columns already present in df:
        trial_num  : total responsive trials (left + right)
        switch_num : number of rule switches
        motor_bias : |median_rt_left - median_rt_right|

    Adds:
        meets_criteria : bool

    Prints summary matching MATLAB output.
    """
    df = df.copy()

    crit1 = df["trial_num"] > NUM_TRIAL_CRITERION
    crit2 = df["switch_num"] > NUM_SWITCH_CRITERION
    df["meets_criteria"] = (crit1 & crit2)

    n_total = len(df)
    n_pass  = df["meets_criteria"].sum()
    print(f"Out of {n_total} sessions, {n_pass} met performance criteria.")

    return df