# reversal_expectation

Python port of the MATLAB two-armed bandit reversal task analysis pipeline (Atilgan & Kwan lab).
Covers preprocessing through behavioral analysis — no plotting.

---

## Structure

```
reversalExpectation/
├── master_bandit.py       # main script — run this
├── behavior/
│   ├── trial_processing.py      # value_getTrialMasks.m + value_getTrialStats.m
│   ├── trial_stats_more.py      # value_getTrialStatsMore.m
│   ├── merge_sessions.py        # merge_sessions.m
│   ├── beh_performance.py       # beh_performance.m
│   ├── choice_switch.py         # choice_switch_hrside.m + variants
│   ├── logistic_regression.py   # logreg_RCUC.m + logreg_RCUC_LR.m
│   ├── changepoint_probability.py
│   └── trial_type_stats.py      # get_lickrate_byTrialType.m + get_val_byTrialType.m
├── preprocessing/
│   ├── presentation_codes.py    # value_getPresentationCodes.m
│   ├── log_parser.py            # parseLogfile.m + value_getSessionData.m
│   ├── lesion_index.py          # addIndexLesion.m + determineBehCriteria.m
│   └── flip_trial_data.py       # fliptrialData.m
└── data/
    └── data-behavior/
        └── bandit_R71_lesion/
            └── data/            # per-animal folders with *.log session files
```

---

## Usage

Run from the project root:

```
python master_bandit.py
```

Output is saved automatically to `analysis/bandit_R71_lesion.csv`.

---

## Field reference

### `get_trial_stats()` output

| Field | Description |
|---|---|
| `c` | Choice: −1 = left, +1 = right, NaN = miss |
| `r` | Outcome: 1 = reward, 0 = no-reward, NaN = miss |
| `rule` | Rule index (1 … n_rules) |
| `rewardprob` | (nTrials, 2) array — [left_prob, right_prob] |

### `get_trial_stats_more()` additions

| Field | Description |
|---|---|
| `blockLength` | Trials per block |
| `blockRule` | Rule index of each block |
| `blockTrans` | Rule index of the next block (NaN for last) |
| `blockTrialtoCrit` | Trials until 10th correct choice |
| `blockTrialRandomAdded` | blockLength − blockTrialtoCrit |
| `hitrates` | Fraction of non-miss trials on hr side |
| `rewardrates` | Mean reward per trial |
| `pWinStay` | P(stay \| rewarded) |
| `pLooseSwitch` | P(switch \| unrewarded) |
| `blockPreSwitchBetterChoiceAtSwitch` | Last choice = hr side (0/1) |
| `hr_side` | Trial-level: −1 = left is hr, +1 = right is hr |
| `ruletransList` | (nTrans, 2) unique [from, to] rule pairs |
