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

## Beyond the behavioral port

The repo has grown past the original MATLAB port. Everything above still
describes `master_bandit.py` (preprocessing + behavioral analysis); these are
the parts added since.

### Environments

| Env | What runs there |
|---|---|
| `revExp` (Python 3.12) | everything except the GLM-HMM fit — `requirements.txt` / `requirements.lock.txt` |
| `ssm` (conda, numpy < 2) | the GLM-HMM, which needs the zashwood/ssm fork |

### Behavioral models — `behavior/beh_models/`

| File | What it does |
|---|---|
| `hmmGlm.py` | GLM-HMM machinery (Ashwood et al. 2022). CV over K is resumable via `checkpoint=` |
| `run_glmhmm.py` | fits the global model (`ashwood_wsls`, K = 2) and saves `analysis/glmhmm_model_ashwood_wsls_K2.pkl` |
| `glmhmm_decode.py` | decodes states on NEW sessions with the frozen model; smoothed and causal (filtered) posteriors |
| `glmhmm_heldout_validation.py` | validates the frozen model on the imaging animals, which were never fit |
| `bayesian_models.py` | belief / belief-CK models (Murphy et al. 2024) |
| `belief_vhr.py` | belief model with a variable hazard H(tau) = sigmoid(a + b*tau); `belief_trajectory()` returns the latents for use as neural regressors |
| `belief_vhr_lrandom.py` | the anticipation test (b > 0) restricted to the pre-switch L_Random window |

### Neuromodulator pipeline — `neuromodulator/`

GRAB-NE and GRAB-ACh miniscope recordings from PRELIMBIC cortex, separate
cohorts (NE: 1930x / 2033x; ACh: 1940x), one ROI per session.

| File | What it does |
|---|---|
| `wang2025_pipeline.py` | replication of Wang et al. 2025 (dF/F, alignment, per-bin regression). `python wang2025_pipeline.py` runs a self-test |
| `neuromodulator_trials.py` | per-trial table + explicit trial <-> imaging-stamp matching; `EXCLUDED_ANIMALS` |
| `verify_neuromodulator_alignment.py` | session/file matching, trial alignment, clock drift |
| `neuromodulator_exploratory.py` | QC, outcome PSTHs, and a NEURAL test of the matching (lag test) |
| `run_wang2025_neuromodulator.py` | the Wang regression over every session, NE vs ACh |
| `neuromodulator_state_regression.py` | nested regressions: Wang base -> + GLM-HMM state -> + belief / tau |

### Known gaps

- `RECORDING_SITE` in `run_wang2025_neuromodulator.py` is empty: the injected
  hemisphere is not recorded anywhere in the data or in the original MATLAB
  code. Until it is filled, C(n) is coded right = 1 / left = 0, NOT ipsi/contra,
  so the SIGN of the choice coefficients is not interpretable.
- Animal 19303 is excluded from the neural analyses (see `neuromodulator_trials.py`).

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
| `blockTrialtoCrit` | Trials until the 10th cumulative choice of the hr side; NaN in the LAST block of each session (never switched, so it is right-censored, as in the .m) |
| `blockTrialRandomAdded` | blockLength − blockTrialtoCrit |
| `hitrates` | PERCENT of hr-side choices over all trials of the block (misses included in the denominator) plus the first trial of the next block (off-by-one of the .m, replicated) |
| `rewardrates` | PERCENT of rewarded trials, same window as `hitrates` |
| `pWinStay` | P(stay \| rewarded), over the LAST 6 trials of the block only |
| `pLooseSwitch` | P(switch \| unrewarded), over the LAST 6 trials of the block only |
| `blockPreSwitchBetterChoiceAtSwitch` | Last choice = hr side (0/1) |
| `hr_side` | Trial-level: −1 = left is hr, +1 = right is hr |
| `ruletransList` | (nTrans, 2) unique [from, to] rule pairs |
