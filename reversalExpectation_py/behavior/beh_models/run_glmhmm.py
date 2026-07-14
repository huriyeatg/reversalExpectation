"""
run_glmhmm.py
=============
Driver script for the Ashwood-style GLM-HMM analysis of the bandit data.

Run this from the `ssm` conda environment (where the zashwood/ssm fork is
installed). It imports hmmGlm.py DIRECTLY (bypassing behavior/__init__.py, so
none of the rest of the revExp pipeline needs to be importable here) and uses
the behaviour CSV produced by the revExp pipeline as the data interface.

    conda activate ssm
    conda install -c conda-forge pandas joblib matplotlib    # one-time, if not present
    python run_glmhmm.py

Compute note: a single global GLM-HMM fit costs ~1 min per animal-worth of data
(~17k trials, 40 EM iters). The full 616-session dataset is a batch/cluster job.
Start with SUBSET_N_ANIMALS set to a small number to gauge cost, then scale.

K selection: held-out LL plateaus after K=2 (the K=2->K=3 and K=3->K=4 gains
both fall below the fold-to-fold noise ~0.009 LL/trial), so K is NOT chosen on
CV alone. The final model is K=3 on interpretability + state-stability grounds;
the RUN_KSELECTION block produces the figures that back that up.

RUN_ANTICIPATION tests whether animals anticipate the switch: P(choose the worse
option | tau) within the exogenous L_Random window, inside the engaged state,
compared against a yoked reactive forward-simulation of the fitted GLM-HMM.
"""

import os
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        module="ssm.optimizers")

# Windows + joblib + ssm/BLAS: cap each process to 1 BLAS thread so that
# parallel workers do not oversubscribe cores and deadlock. MUST run before
# numpy is imported to take effect.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

# --- make hmmGlm.py + the analysis modules importable directly --------------
# (no behavior/__init__.py side effects). This script lives in beh_models/
# alongside the modules it imports, so it resolves its own location instead of
# hardcoding an absolute path -- portable across clones / machines.
HERE = Path(__file__).resolve().parent          # .../behavior/beh_models
REPO_ROOT = HERE.parents[1]                      # .../reversalExpectation_py
sys.path.insert(0, str(HERE))
import hmmGlm as g                 # noqa: E402
import glmhmm_kselection as ks     # noqa: E402
import anticipation_test as at     # noqa: E402
import glmhmm_occupancy as occ     # noqa: E402

# REWARD_PROBS is the task's rule -> (p_left, p_right) table. Import it from the
# pipeline; fall back to the 2-rule 70:10 map (VERIFY the rule indices if used).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from preprocessing.presentation_codes import REWARD_PROBS   # noqa: E402
except Exception as _e:                                          # pragma: no cover
    print(f"[anticipation] could not import REWARD_PROBS ({_e}); "
          "using hardcoded 2-rule 70:10 map — VERIFY the rule indices.")
    REWARD_PROBS = {2: {0: (0.70, 0.10), 1: (0.10, 0.70)}}


# ===========================================================================
# Config
# ===========================================================================
CSV_PATH   = REPO_ROOT / "analysis" / "bandit_R71_lesion.csv"
OUTPUT_DIR = REPO_ROOT / "analysis"   # CSVs / data
FIGS_DIR   = REPO_ROOT / "figs"       # all images

PARAMETRIZATION = "ashwood_wsls"   # "ashwood_wsls" (faithful) | "reward_perseveration"

# Subset for exploration. None = all selected sessions (full batch run).
SUBSET_N_ANIMALS = None

K_RANGE     = (1, 2, 3, 4)   # swept by the CV (needed for the K-selection curve)
CHOSEN_K    = 3              # final model + state-stability (must be in K_RANGE)
N_RESTARTS  = 15
N_ITERS     = 200
N_FOLDS     = 5
N_JOBS      = -1             # parallel workers for CV / comparison / final fit (n_jobs=-1 uses all cores)

RUN_CV           = False     # cross-validate held-out LL across K
RUN_COMPARISON   = False     # GLM vs lapse vs GLM-HMM(CHOSEN_K)
RUN_KSELECTION   = False     # slide figures: CV curve + state-stability
RUN_ANTICIPATION = True     # anticipation test (P(worse|tau) in L_Random)
RUN_STATE_OCCUPANCY = False  # state occupancy: block-end aligned + within-session
RUN_PER_SESSION_OCCUPANCY = True  # one figure per session (~616 files); slow, off by default

# FORCE_REFIT: if True, always re-fit the GLM-HMM and OVERWRITE glmhmm_states.csv,
# even when that cache already exists. Use it whenever the model or the data
# changed, so the cached per-trial states never go stale. If False (default),
# RUN_STATE_OCCUPANCY reuses an existing glmhmm_states.csv and only fits when the
# cache is missing (see the cache-or-fit rule in the State occupancy block).
FORCE_REFIT = False

# --- state-stability sub-settings (only used when RUN_KSELECTION) -----------
N_STABILITY_RESTARTS = 10
RUN_FOLD_STABILITY   = False
# State labels are chosen by model size and are POSITIONAL (state 0 = the most
# perseverative state, which is how the fit has consistently ordered them).
# Always cross-check against the printed per-state weights; if a fit ever
# reorders the states, adjust the mapping here.
_STATE_LABELS_BY_K = {2: ["exploit", "explore"],
                      3: ["engaged", "random", "side-biased"]}
STATE_LABELS = _STATE_LABELS_BY_K.get(CHOSEN_K, [f"state {i}" for i in range(CHOSEN_K)])

# --- anticipation sub-settings (only used when RUN_ANTICIPATION) ------------
ANTICIPATION_MAX_TAU = 14    # how far into L_Random to plot
N_ANTICIPATION_SIMS  = 200   # yoked reactive forward sims (pure-Python; lower if slow)


# ===========================================================================
# Helpers
# ===========================================================================
def _fold_training_dfs(df, n_folds, seed=0):
    """Per-fold training DataFrames, sessions split stratified by animal."""
    rng = np.random.default_rng(seed)
    fold_of = {}
    for animal, sub in df.groupby("animal", sort=False):
        sessions = list(dict.fromkeys(zip(sub["animal"], sub["session_file"])))
        order = rng.permutation(len(sessions))
        for rank, idx in enumerate(order):
            fold_of[sessions[idx]] = rank % n_folds
    keys = list(zip(df["animal"], df["session_file"]))
    fold_col = np.array([fold_of[k] for k in keys])
    return [df[fold_col != f].copy() for f in range(n_folds)]


def attach_glmhmm_states(df, post, tags, engaged_idx):
    """
    Map per-session posteriors (kept-session list, in tag order) back onto df
    rows, adding 'glmhmm_state' (hard MAP) and 'p_engaged'. Sessions skipped by
    build_glmhmm_inputs (fewer than min_trials observed) get NaN and are dropped
    downstream. Alignment is by (animal, session_file) tag, never by blind
    concatenation, because the kept-session list omits the skipped sessions.
    """
    state = pd.Series(np.nan, index=df.index)
    p_eng = pd.Series(np.nan, index=df.index)
    for (animal, ses), P in zip(tags, post):
        rows = df.index[(df["animal"] == animal) & (df["session_file"] == ses)]
        if len(rows) != P.shape[0]:
            raise ValueError(f"length mismatch for {(animal, ses)}: "
                             f"{len(rows)} df rows vs {P.shape[0]} posterior rows")
        state.loc[rows] = P.argmax(1)
        p_eng.loc[rows] = P[:, engaged_idx]
    return df.assign(glmhmm_state=state, p_engaged=p_eng)


# ===========================================================================
# Main
# ===========================================================================
def main():
    df = pd.read_csv(CSV_PATH)
    df = g.select_glmhmm_sessions(df)

    if SUBSET_N_ANIMALS is not None:
        top = (df.groupby("animal")["session_file"].nunique()
                 .sort_values(ascending=False).head(SUBSET_N_ANIMALS).index.tolist())
        df = df[df["animal"].isin(top)].copy()
        print(f"[subset] {SUBSET_N_ANIMALS} animals: {top} -> "
              f"{df.groupby(['animal','session_file']).ngroups} sessions")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- decide whether the final GLM-HMM fit is needed ------------------
    # The fit produces `model` (per-state weights + transition matrix), which
    # the anticipation test and the state-stability figures cannot do without.
    # State occupancy needs per-trial states too, but those can be reused from a
    # cached glmhmm_states.csv -- so occupancy alone triggers a fit ONLY when
    # that cache is missing or FORCE_REFIT is set. If the fit runs for ANY
    # reason, occupancy uses the fresh model and refreshes the cache.
    states_csv = OUTPUT_DIR / f"glmhmm_states_K{CHOSEN_K}.csv"
    occupancy_needs_fit = RUN_STATE_OCCUPANCY and (FORCE_REFIT or not states_csv.exists())
    need_model = RUN_ANTICIPATION or RUN_KSELECTION or occupancy_needs_fit

    cv = None
    if RUN_CV:
        print("\n=== Cross-validation over K ===")
        t0 = time.time()
        cv = g.cross_validate_glmhmm(df, K_range=K_RANGE, parametrization=PARAMETRIZATION,
                                     n_folds=N_FOLDS, n_restarts=N_RESTARTS,
                                     n_iters=N_ITERS, n_jobs=N_JOBS)
        cv.to_csv(OUTPUT_DIR / "glmhmm_cv.csv", index=False)
        print(f"\nCV done in {time.time()-t0:.0f}s. Mean held-out LL/trial by K:")
        print(cv.groupby("K")["test_ll_per_trial"].mean().round(4).to_string())
        best_K = int(cv.groupby("K")["test_ll_per_trial"].mean().idxmax())
        print(f"idxmax K (NB: gains beyond K=2 are within fold noise): {best_K}")

    baselines = None
    if RUN_COMPARISON:
        print("\n=== Model comparison (GLM vs lapse vs GLM-HMM) ===")
        t0 = time.time()
        cmp = g.model_comparison(df, K_states=CHOSEN_K, parametrization=PARAMETRIZATION,
                                 n_folds=N_FOLDS, n_restarts=N_RESTARTS,
                                 n_iters=N_ITERS, n_jobs=N_JOBS)
        cmp.to_csv(OUTPUT_DIR / "glmhmm_model_comparison.csv", index=False)
        print(f"\nComparison done in {time.time()-t0:.0f}s. Mean held-out LL/trial:")
        means = cmp.groupby("model")["test_ll_per_trial"].mean()
        print(means.round(4).to_string())
        baselines = {}
        for name in ("GLM", "lapse"):
            hit = [m for m in means.index if name.lower() in str(m).lower()]
            if hit:
                baselines[name] = float(means[hit[0]])
        baselines = baselines or None

    if RUN_KSELECTION and cv is not None and cv["K"].nunique() > 1:
        fig_cv = ks.plot_cv_curve(
            cv, chosen_K=CHOSEN_K, baselines=baselines, n_folds=N_FOLDS,
            ll_col="test_ll_per_trial",
            outfile=str(FIGS_DIR / "cv_kselection.png"))
        print(f"\n[kselection] wrote {fig_cv}")

    # ---- final fit (runs only if something needs the model) ---------------
    did_fit = False
    if need_model:
        # inputs are needed only for the fit / stability / anticipation blocks
        ch, inp, mk, tags = g.build_glmhmm_inputs(df, parametrization=PARAMETRIZATION)
        print(f"\n=== Final fit (all selected sessions, K={CHOSEN_K}) ===")
        _, glm_w = g.fit_glm(ch, inp, mk, n_jobs=N_JOBS)
        model = g.fit_global_glmhmm(ch, inp, mk, K=CHOSEN_K, glm_weights=glm_w,
                                    n_restarts=N_RESTARTS, n_iters=N_ITERS, n_jobs=N_JOBS)
        weights_df = g.glmhmm_weights(model, parametrization=PARAMETRIZATION)
        print("per-state weights (log-odds of choosing right):")
        print(weights_df.round(3).to_string(index=False))
        print("\ntransition matrix:")
        print(g.glmhmm_transition_matrix(model).round(3))
        # engaged = the state with the strongest perseveration (largest prev_choice)
        engaged_idx = int(weights_df.set_index("state")["prev_choice"].idxmax())
        print(f"engaged state index = {engaged_idx}")
        did_fit = True

    # ---- state occupancy: CACHE-OR-FIT ------------------------------------
    # Exactly what this block does, depending on whether the fit ran above:
    #   * did_fit == True  (model is fresh): derive per-trial states from the
    #     fitted model and (OVER)WRITE glmhmm_states.csv, then plot. This both
    #     refreshes the cache and uses up-to-date states.
    #   * did_fit == False (fit was skipped because a valid glmhmm_states.csv
    #     exists, FORCE_REFIT is False, and nothing else needed the model):
    #     LOAD the cached states, merge them onto the filtered bandit df by
    #     (animal, session_file, trial_idx), and plot. No fitting -- seconds.
    if RUN_STATE_OCCUPANCY:
        print("\n=== State occupancy (block-end aligned + within-session) ===")
        if did_fit:
            print("  [fit path] deriving per-trial states from the fitted model "
                  "and writing glmhmm_states.csv")
            post = g.glmhmm_posteriors(model, ch, inp, mk)
            dfo = attach_glmhmm_states(df, post, tags, engaged_idx)
            dfo = dfo[dfo["glmhmm_state"].notna()].copy()
            keep = [c for c in ["animal", "session_file", "block_idx", "trial_idx",
                                "glmhmm_state", "p_engaged"] if c in dfo.columns]
            dfo[keep].to_csv(states_csv, index=False)
            print(f"  wrote {states_csv}")
        else:
            print(f"  [cache path] reusing {states_csv.name} -- GLM-HMM fit SKIPPED. "
                  "Set FORCE_REFIT=True to re-fit and overwrite it.")
            states = pd.read_csv(states_csv)
            merge_keys = ["animal", "session_file", "trial_idx"]
            state_cols = [c for c in ["glmhmm_state", "p_engaged"] if c in states.columns]
            missing = [k for k in merge_keys if k not in states.columns]
            if missing:
                raise ValueError(f"{states_csv.name} lacks merge keys {missing}; "
                                 "re-run with FORCE_REFIT=True to regenerate it.")
            dfo = df.merge(states[merge_keys + state_cols], on=merge_keys, how="inner")
            dfo = dfo[dfo["glmhmm_state"].notna()].copy()
            print(f"  merged cached states onto {len(dfo)} trials, "
                  f"{dfo.groupby(['animal','session_file']).ngroups} sessions")
        OCC = FIGS_DIR / "occupancy"          # group all occupancy figures here
        fig_o = occ.plot_state_occupancy(
            dfo, state_labels=STATE_LABELS,
            outfile=str(OCC / f"glmhmm_state_occupancy_K{CHOSEN_K}.png"))
        print(f"[occupancy] wrote {fig_o}")

        # L_Random-stratified (pre-switch only), by-tau, and across-switch views
        occ.plot_state_occupancy_by_lrandom(
            dfo, state_labels=STATE_LABELS,
            outfile=str(OCC / f"glmhmm_state_occupancy_by_lrandom_K{CHOSEN_K}.png"))
        occ.plot_state_occupancy_by_lrandom_tau(
            dfo, state_labels=STATE_LABELS,
            outfile=str(OCC / f"glmhmm_state_occupancy_tau_K{CHOSEN_K}.png"))
        occ.plot_state_occupancy_by_lrandom_tau(
            dfo, state_labels=STATE_LABELS, normalize_by_animal=True,
            outfile=str(OCC / f"glmhmm_state_occupancy_tau_norm_K{CHOSEN_K}.png"))
        occ.plot_state_occupancy_across_switch(
            dfo, state_labels=STATE_LABELS,
            outfile=str(OCC / f"glmhmm_state_occupancy_across_switch_K{CHOSEN_K}.png"))
        print(f"[occupancy] wrote stratified / tau / across-switch figures to {OCC}")

        # one figure per session (many files) -> figs/occupancy/per_session/
        if RUN_PER_SESSION_OCCUPANCY:
            occ.plot_per_session_occupancy(
                dfo, state_labels=STATE_LABELS,
                outdir=str(OCC / "per_session"))

    # ---- state-stability (slide) ------------------------------------------
    if RUN_KSELECTION:
        reg_names = [c for c in weights_df.columns if c != "state"]

        def fit_one(ch_, inp_, mk_, K, seed):
            return g.fit_global_glmhmm(ch_, inp_, mk_, K=K, glm_weights=glm_w,
                                       n_restarts=1, n_iters=N_ITERS,
                                       seed=seed, n_jobs=1)

        def weights_of(m):
            return g.glmhmm_weights(m, parametrization=PARAMETRIZATION)

        transmat_of = g.glmhmm_transition_matrix

        print(f"\n=== State stability: {N_STABILITY_RESTARTS} restarts (K={CHOSEN_K}) ===")
        t0 = time.time()
        stats_r = ks.collect_fits([(ch, inp, mk)], K=CHOSEN_K,
                                  seeds=range(N_STABILITY_RESTARTS),
                                  fit_fn=fit_one, weights_fn=weights_of,
                                  transmat_fn=transmat_of)
        print(f"done in {time.time()-t0:.0f}s")
        print(" state norms      :", [round(v, 2) for v in stats_r["state_norm"]])
        print(" matched cosine   : min(struct)="
              f"{stats_r['overall_min_cos_structured']:.3f}, "
              f"max weight SD={stats_r['max_weight_sd']:.3f}")
        print(" dwell mean       :", [round(v, 1) for v in stats_r["dwell_mean"]])
        fig_r = ks.plot_state_stability(
            stats_r, reg_names, state_labels=STATE_LABELS,
            outfile=str(FIGS_DIR / f"state_stability_restarts_K{CHOSEN_K}.png"))
        print(f"[kselection] wrote {fig_r}")

        if RUN_FOLD_STABILITY:
            print(f"\n=== State stability: {N_FOLDS} folds (K={CHOSEN_K}) ===")
            t0 = time.time()
            train_dfs = _fold_training_dfs(df, N_FOLDS, seed=0)
            fold_sets = []
            for tdf in train_dfs:
                c, i, m, _ = g.build_glmhmm_inputs(tdf, parametrization=PARAMETRIZATION)
                fold_sets.append((c, i, m))
            stats_f = ks.collect_fits(fold_sets, K=CHOSEN_K,
                                      seeds=[0] * len(fold_sets),
                                      fit_fn=fit_one, weights_fn=weights_of,
                                      transmat_fn=transmat_of)
            print(f"done in {time.time()-t0:.0f}s")
            print(" matched cosine   : min(struct)="
                  f"{stats_f['overall_min_cos_structured']:.3f}, "
                  f"max weight SD={stats_f['max_weight_sd']:.3f}")
            fig_f = ks.plot_state_stability(
                stats_f, reg_names, state_labels=STATE_LABELS,
                outfile=str(FIGS_DIR / f"state_stability_folds_K{CHOSEN_K}.png"))
            print(f"[kselection] wrote {fig_f}")

    # ---- anticipation test ------------------------------------------------
    if RUN_ANTICIPATION:
        print("\n=== Anticipation test (P(worse|tau) in L_Random) ===")
        t0 = time.time()
        # 1) per-trial GLM-HMM state, mapped back to df rows by tag
        post = g.glmhmm_posteriors(model, ch, inp, mk)
        dfa = attach_glmhmm_states(df, post, tags, engaged_idx)
        dfa = dfa[dfa["glmhmm_state"].notna()].copy()
        dfa = at.prepare_anticipation(dfa, reward_probs=REWARD_PROBS)

        # 2) data curves: all states, engaged only, and P(engaged|tau)
        cur_all = at.pworse_curve(dfa, max_tau=ANTICIPATION_MAX_TAU)
        cur_eng = at.pworse_curve(dfa, state_col="glmhmm_state",
                                  state_value=engaged_idx, max_tau=ANTICIPATION_MAX_TAU)
        peng = at.pstate_curve(dfa, state_col="glmhmm_state",
                               target_value=engaged_idx, max_tau=ANTICIPATION_MAX_TAU)

        # 3) reactive null: yoked forward sim of the fitted GLM-HMM, engaged-only band
        Wr = weights_df[["bias", "prev_choice", "wsls"]].to_numpy()
        print(f"  simulating {N_ANTICIPATION_SIMS} reactive runs (pure-Python; "
              "this can take a few minutes on the full dataset)...")
        c_sim, z_sim = at.simulate_reactive_glmhmm(
            dfa, Wr, g.glmhmm_transition_matrix(model),
            n_sims=N_ANTICIPATION_SIMS, rng=np.random.default_rng(0),
            reward_probs=REWARD_PROBS)
        band_eng = at.reactive_pworse_band(dfa, c_sim, z_sim=z_sim,
                                           state_value=engaged_idx,
                                           max_tau=ANTICIPATION_MAX_TAU)

        # 4) a one-line numeric read (excess of engaged data over null)
        ok = cur_eng["enough"].to_numpy() & ~np.isnan(band_eng["mean"].to_numpy())
        if ok.sum() > 2:
            excess = (cur_eng["p"] - band_eng["mean"]).to_numpy()[ok]
            taus = cur_eng["tau"].to_numpy()[ok]
            slope = float(np.polyfit(taus, excess, 1)[0])
            print(f"  engaged excess over reactive null: mean={np.nanmean(excess):+.3f}, "
                  f"slope={slope:+.4f}/trial  (>0 = anticipation)")

        # 5) inference on the excess slope: animal-bootstrap 95% CI + two-sided
        #    p, and the per-animal slope (subject-level replication, analogous to
        #    the per-animal hazard slope b of belief_vhr). Reuses c_sim / z_sim,
        #    so it adds no simulation cost.
        inf = at.anticipation_inference(
            dfa, c_sim, z_sim, state_value=engaged_idx,
            max_tau=ANTICIPATION_MAX_TAU, n_boot=2000,
            rng=np.random.default_rng(0))
        inf["per_animal"].to_csv(OUTPUT_DIR / f"anticipation_per_animal_K{CHOSEN_K}.csv", index=False)
        print(f"  wrote {OUTPUT_DIR / f'anticipation_per_animal_K{CHOSEN_K}.csv'}")

        # 6) figure, with the inference annotated on the anticipation panel
        fig_a = at.plot_anticipation(cur_all, cur_eng, band_eng, pstate=peng,
                                     state_name=STATE_LABELS[engaged_idx], inference=inf,
                                     outfile=str(FIGS_DIR / f"anticipation_test_K{CHOSEN_K}.png"))
        print(f"done in {time.time()-t0:.0f}s")
        print(f"[anticipation] wrote {fig_a}")


if __name__ == "__main__":
    main()