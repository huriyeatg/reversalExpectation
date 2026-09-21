"""
add_index_neuromodulator.py
===========================
Port of addIndexNeuromodulator.m (H Atilgan & AC Kwan, 20092020), with the
session <-> miniscope file matching rewritten.

addIndexNeuromodulator.m rule: dir('M{Animal}_Phase{Phase}_{yymmdd}*_cell.csv').
Weak points of that rule (and of the previous Python port):
  (1) if date_number is missing the prefix has no date and EVERY session gets
      the animal's first file, silently;
  (2) with two files on the same day, MATLAB fails ({filepath.name} has >1
      element) and the previous port silently took the first one alphabetically;
  (3) off-pattern names (M20330_2_..., N20330_...) never match;
  (4) nothing prevents one file from being assigned to several sessions.

Current rule:
  - Session date/time comes from the .log name (..._yymmddHHMM), not from
    date_number.
  - Each file's date/time comes from its name (..._yymmddHHMMSS_cell.csv).
  - One-to-one matching within the same animal, closest pairs first,
    with |dt| <= TOL_MIN minutes.
  - timeEvents: same prefix as the assigned cell file; otherwise by timestamp.
  - Files prefixed with "X_" are excluded (flagged as discarded).
In the normal case (one file per session, same day) it gives the same result
as the .m rule; verify_neuromodulator_alignment.py checks this per session.

Output columns (same as before + 2 diagnostics):
    experiment           : 1 = norepinephrine, 2 = acetylcholine (3rd digit of the ID)
    cell_created         : 1.0 if a *_cell.csv was assigned, else NaN
    time_events_created  : 1.0 if a *_timeEvents.csv was assigned, else NaN
    cell_filename        : assigned trace file ("" if none)
    time_events_filename : assigned timeEvents file ("" if none)
    neural_data_path     : miniscope folder
    cell_dt_min          : minutes between cell start and log start (diagnostic)
    neural_match_note    : "" / "no_cell_within_tol" / "te_by_timestamp" / "no_te"
"""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TOL_MIN = 120                   # tolerance (validated: median dt = +2 min, max |dt| = 86)
EXCLUDE_PREFIXES = ("X_",)

_NEURAL_RE = re.compile(
    r"^(?P<prefix>[A-Za-z_]*?)(?P<animal>\d{4,5})(?:_\d+)?_[Pp]hase(?P<phase>\d+)_"
    r"(?P<ts>\d{12})_(?P<kind>cell|timeEvents)\.csv$", re.IGNORECASE)
_BEH_TS_RE = re.compile(r"_(\d{10})$")


def _experiment_from_animal(animal: str) -> float:
    """3rd digit of the ID: '3' -> NE (1), '4' -> ACh (2). Same as the .m."""
    if len(animal) >= 3:
        return {"3": 1.0, "4": 2.0}.get(animal[2], np.nan)
    return np.nan


def _scan_neural_files(signal_path: Path) -> pd.DataFrame:
    rows = []
    for p in signal_path.glob("*.csv"):
        m = _NEURAL_RE.match(p.name)
        if not m:
            continue
        pref = m.group("prefix")
        if pref.upper().startswith(tuple(x.upper() for x in EXCLUDE_PREFIXES)):
            continue
        rows.append({
            "name": p.name, "animal": m.group("animal"),
            "t": datetime.strptime(m.group("ts"), "%y%m%d%H%M%S"),
            "kind": "cell" if m.group("kind").lower() == "cell" else "timeEvents",
            "stem": p.name[: m.end("ts")],
        })
    return pd.DataFrame(rows, columns=["name", "animal", "t", "kind", "stem"])


def _greedy_match(sess_t: pd.Series, sess_animal: pd.Series,
                  files: pd.DataFrame, tol_min: float) -> dict:
    """One-to-one, closest pairs first. Returns {session_idx: (file_idx, dt)}."""
    pairs = []
    for i in sess_t.index:
        if pd.isna(sess_t[i]):
            continue
        cand = files[files["animal"] == sess_animal[i]]
        dts = (cand["t"] - sess_t[i]).dt.total_seconds() / 60
        for j, dt in dts[dts.abs() <= tol_min].items():
            pairs.append((abs(dt), dt, i, j))
    pairs.sort()
    used_s, used_f, out = set(), set(), {}
    for _, dt, i, j in pairs:
        if i not in used_s and j not in used_f:
            used_s.add(i); used_f.add(j)
            out[i] = (j, dt)
    return out


def add_index_neuromodulator(data_index: pd.DataFrame, signal_path: str,
                             tol_min: float = TOL_MIN,
                             verbose: bool = True) -> pd.DataFrame:
    signal_path = Path(signal_path)
    files = _scan_neural_files(signal_path)
    cells = files[files["kind"] == "cell"]
    tes = files[files["kind"] == "timeEvents"]
    te_by_stem = dict(zip(tes["stem"], tes["name"]))

    animal = data_index["animal"].astype(str)
    stem = data_index["session_file"].map(lambda s: Path(str(s)).stem)
    ts = stem.str.extract(_BEH_TS_RE)[0]
    sess_t = pd.to_datetime(ts, format="%y%m%d%H%M", errors="coerce")

    m_cell = _greedy_match(sess_t, animal, cells, tol_min)
    m_te = _greedy_match(sess_t, animal, tes, tol_min)      # fallback

    records = []
    for i in data_index.index:
        rec = {
            "experiment": _experiment_from_animal(animal[i]),
            "cell_created": np.nan, "time_events_created": np.nan,
            "cell_filename": "", "time_events_filename": "",
            "neural_data_path": str(signal_path),
            "cell_dt_min": np.nan, "neural_match_note": "",
        }
        if i not in m_cell:
            rec["neural_match_note"] = "no_cell_within_tol"
        else:
            j, dt = m_cell[i]
            rec.update(cell_created=1.0, cell_filename=cells.loc[j, "name"],
                       cell_dt_min=round(dt, 1))
            te_name = te_by_stem.get(cells.loc[j, "stem"])
            if te_name is None and i in m_te:
                te_name = tes.loc[m_te[i][0], "name"]
                rec["neural_match_note"] = "te_by_timestamp"
            if te_name is None:
                rec["neural_match_note"] = "no_te"
            else:
                rec.update(time_events_created=1.0, time_events_filename=te_name)
        records.append(rec)

    nm_df = pd.DataFrame(records, index=data_index.index)
    # if data_index already had these columns (re-run), replace them
    base = data_index.drop(columns=[c for c in nm_df.columns if c in data_index.columns])
    out = pd.concat([base, nm_df], axis=1)

    if verbose:
        n_ok = int(out["cell_created"].notna().sum())
        big = out[out["cell_dt_min"].abs() > 10]
        print(f"[neural match] {n_ok}/{len(out)} sessions with their own trace "
              f"(tolerance ±{tol_min} min); "
              f"{int(out['time_events_created'].notna().sum())} with timeEvents; "
              f"{len(big)} with |dt| > 10 min")
    return out
