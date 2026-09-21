"""
glmhmm_heldout_validation.py
============================
Validates the global GLM-HMM (ashwood_wsls, K=2) on data it was NOT fit on:
the neuromodulator imaging sessions, i.e. 13 NEW animals performing the same
task (rules 41/42, 70:10). The model is frozen (no refitting).

(A) Predictive validation -- held-out log-likelihood per trial:
      GLM-HMM (K=2)  vs  1-state GLM  vs  chance (log 0.5)
    per session and per animal; paired Wilcoxon across animals.
    This mirrors Ashwood et al.'s model comparison, but on new animals rather
    than held-out sessions of the training animals.

(B) Behavioral validation -- the three read-outs used on the training data,
    recomputed on the new animals with states decoded by the frozen model:
      V1  P(chose better | exploit) vs P(chose better | explore), pre-switch
          L_Random window (complete blocks, last block excluded)
      V2  P(explore) vs trials since block switch (pre-criterion trials)
      V3  median reaction time, exploit vs explore
    All tested per animal (paired Wilcoxon), animal = unit of replication.

(C) Comparability: the same session inclusion rule as the training set
    (responded trials > 100 and switches > 3) is applied, and basic behavioral
    metrics are printed next to the training dataset's for comparison.

Requires analysis/glmhmm_model_ashwood_wsls_K2.pkl (run_glmhmm.py with the
model-saving patch). Run from behavior/beh_models/ in the `ssm` env:
    conda activate ssm
    python glmhmm_heldout_validation.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))
import hmmGlm as g  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dec = _load(HERE / "glmhmm_decode.py", "glmhmm_decode")
NM = REPO_ROOT / "neuromodulator"
ntr = _load(NM / "neuromodulator_trials.py", "neuromodulator_trials")

MODEL_PKL = REPO_ROOT / "analysis" / "glmhmm_model_ashwood_wsls_K2.pkl"
TRAIN_CSV = REPO_ROOT / "analysis" / "bandit_R71_lesion.csv"
OUT_DIR = REPO_ROOT / "analysis" / "glmhmm_heldout"
PHASES = [31]
MIN_RESPONDED, MIN_SWITCHES = 100, 3        # same rule as lesion_index.compute_session_criteria
MIN_TRIALS_STATE = 20                       # per animal and state, for V1 / V3
POS_BINS = [0, 3, 6, 10, 15, 25, 200]       # V2 bins of trials since switch
COHORT = {1.0: "NE", 2.0: "ACh"}


# ---------------------------------------------------------------------------
def session_loglik(model, df_ses, bundle):
    ch, X, mk = g.build_session_arrays(df_ses[["choice", "rewarded"]],
                                       n_lags=bundle["n_lags"],
                                       parametrization=bundle["parametrization"])
    n_obs = int(mk.sum())
    ll = float(model.log_likelihood([ch], inputs=[X], masks=[mk]))
    return ll, n_obs


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if not len(d):
        return dict(n=0, median_diff=np.nan, frac_pos=np.nan, p=np.nan)
    # Wilcoxon needs at least 5 pairs to be meaningful; below that, report the
    # effect only
    p = float(stats.wilcoxon(d).pvalue) if len(d) >= 5 else np.nan
    return dict(n=int(len(d)), median_diff=float(np.median(d)),
                frac_pos=float(np.mean(d > 0)), p=p)


def load_sessions():
    master = _load(NM / "master_neuromodulator.py", "master_neuromodulator")
    idx = master.build_index()
    idx = idx[idx["phase"].isin(PHASES)]
    tables, meta = [], []
    for _, r in idx.iterrows():
        ses = Path(str(r["session_file"])).stem
        try:
            d = ntr.build_trial_table(r["beh_path"], str(r["animal"]), ses)
        except Exception as e:
            meta.append({"session": ses, "animal": str(r["animal"]), "status": f"error: {e!r}"})
            continue
        n_resp, n_sw = int(d["responded"].sum()), int(d["block_idx"].max())
        keep = (n_resp > MIN_RESPONDED) and (n_sw > MIN_SWITCHES)
        d["cohort"] = COHORT.get(r.get("experiment"), "?")
        meta.append({"session": ses, "animal": str(r["animal"]), "cohort": d["cohort"].iloc[0],
                     "n_responded": n_resp, "n_switches": n_sw,
                     "status": "ok" if keep else "fails_criteria"})
        if keep:
            tables.append(d)
    return tables, pd.DataFrame(meta)


# ---------------------------------------------------------------------------
def part_a(bundle, tables):
    rows = []
    for d in tables:
        ll_h, n = session_loglik(bundle["model"], d, bundle)
        row = {"animal": d["animal"].iloc[0], "cohort": d["cohort"].iloc[0],
               "session": d["session_file"].iloc[0], "n_obs": n,
               "ll_glmhmm": ll_h / n, "ll_chance": np.log(0.5)}
        if bundle.get("glm_model") is not None:
            row["ll_glm"] = session_loglik(bundle["glm_model"], d, bundle)[0] / n
        rows.append(row)
    ses = pd.DataFrame(rows)
    # per-animal means weighted by trials
    agg = {c: (lambda x, c=c: np.average(ses.loc[x.index, c], weights=ses.loc[x.index, "n_obs"]))
           for c in ["ll_glmhmm", "ll_glm", "ll_chance"] if c in ses}
    ani = ses.groupby(["animal", "cohort"]).agg(**{c: (c, f) for c, f in agg.items()}).reset_index()
    return ses, ani


def decode_all(bundle, tables):
    out = []
    for d in tables:
        s = dec.decode_session(bundle, d["choice"].to_numpy(), d["rewarded"].to_numpy())
        out.append(pd.concat([d.reset_index(drop=True),
                              s[["glmhmm_state", "is_exploit", "p_engaged", "p_engaged_filtered"]]], axis=1))
    return pd.concat(out, ignore_index=True)


def part_b(df):
    res = {}
    # V1: accuracy by state in the pre-switch L_Random window
    w = df[df["in_lrandom_window"] & df["responded"]]
    v1 = (w.groupby(["animal", "is_exploit"])["chose_better"].agg(["mean", "size"])
            .reset_index().pivot(index="animal", columns="is_exploit"))
    ok = (v1[("size", 1)] >= MIN_TRIALS_STATE) & (v1[("size", 0)] >= MIN_TRIALS_STATE)
    v1 = pd.DataFrame({"exploit": v1[("mean", 1)], "explore": v1[("mean", 0)]})[ok]
    res["V1"] = (v1, paired(v1["exploit"], v1["explore"]))

    # V2: P(explore) vs trials since switch (pre-criterion trials, not the first block)
    pre = df[(df["tau"] < 0) & (df["block_idx"] > 0)].copy()
    pre["pos_bin"] = pd.cut(pre["pos_in_block"], POS_BINS, right=False)
    v2 = (1 - pre.groupby(["animal", "pos_bin"], observed=True)["is_exploit"].mean()).unstack()
    first, late = v2.columns[0], v2.columns[2]
    res["V2"] = (v2, paired(v2[late], v2[first]))       # P(explore) higher a few trials after the switch?

    # V3: reaction time by state
    rt = df[df["responded"] & (df["rt"] > 0)]
    rt = rt[rt["rt"] <= rt["rt"].quantile(0.995)]
    v3 = rt.groupby(["animal", "is_exploit"])["rt"].agg(["median", "size"]).reset_index()
    v3 = v3[v3["size"] >= MIN_TRIALS_STATE].pivot(index="animal", columns="is_exploit", values="median")
    v3 = v3.dropna().rename(columns={1: "exploit", 0: "explore"})
    res["V3"] = (v3, paired(v3["explore"], v3["exploit"]))
    return res


def training_comparison(meta_ok, tables):
    """Median per-session behavior: held-out imaging cohort vs training CSV."""
    held = pd.DataFrame([ntr.session_summary(d) for d in tables])
    rows = {"held-out (imaging)": held[["reward_rate", "p_better", "miss_rate",
                                        "median_trials_to_crit", "median_lrandom"]].median()}
    if TRAIN_CSV.exists():
        tr = pd.read_csv(TRAIN_CSV, low_memory=False)
        tr = tr[tr["lesioned"].isna() & (tr["meets_criteria"] == True)]
        s = tr.groupby(["animal", "session_file"]).apply(lambda x: pd.Series({
            "reward_rate": x.loc[x["choice"].notna(), "rewarded"].mean(),
            "p_better": (x.loc[x["choice"].notna(), "choice"] == x.loc[x["choice"].notna(), "hr_side"]).mean(),
            "miss_rate": x["choice"].isna().mean(),
            "median_trials_to_crit": x.groupby("block_idx")["block_trial_to_crit"].first().median(),
            "median_lrandom": x.groupby("block_idx")["block_trial_random_added"].first().median()}))
        rows["training (behavior)"] = s.median()
    return pd.DataFrame(rows).T.round(3)


def figure(ani, res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    col = {"NE": "#2A9D8F", "ACh": "#E9A93A"}
    # A
    if "ll_glm" in ani:
        for _, r in ani.iterrows():
            ax[0].plot([0, 1, 2], [r["ll_chance"], r["ll_glm"], r["ll_glmhmm"]], "-o",
                       color=col.get(r["cohort"], "gray"), alpha=0.7, ms=4)
        ax[0].set_xticks([0, 1, 2], ["chance", "GLM", "GLM-HMM"])
    ax[0].set(ylabel="held-out LL / trial", title="A. Prediction on new animals")
    # V1
    v1 = res["V1"][0]
    for _, r in v1.iterrows():
        ax[1].plot([0, 1], [r["exploit"], r["explore"]], "-o", color="gray", alpha=0.6, ms=4)
    ax[1].axhline(0.5, ls=":", color="k")
    ax[1].set_xticks([0, 1], ["exploit", "explore"])
    ax[1].set(ylabel="P(chose better)", title=f"V1 (p={res['V1'][1]['p']:.2g})")
    # V2
    v2 = res["V2"][0]
    x = np.arange(v2.shape[1])
    for _, r in v2.iterrows():
        ax[2].plot(x, r.values, color="gray", alpha=0.3)
    ax[2].plot(x, v2.mean().values, "-o", color="#C9472B", lw=2)
    ax[2].set_xticks(x, [str(c) for c in v2.columns], rotation=30)
    ax[2].set(ylabel="P(explore)", xlabel="trials since switch", title="V2")
    # V3
    v3 = res["V3"][0]
    for _, r in v3.iterrows():
        ax[3].plot([0, 1], [r["exploit"], r["explore"]], "-o", color="gray", alpha=0.6, ms=4)
    ax[3].set_xticks([0, 1], ["exploit", "explore"])
    ax[3].set(ylabel="median RT (s)", title=f"V3 (p={res['V3'][1]['p']:.2g})")
    fig.suptitle("Frozen global GLM-HMM validated on animals not used for fitting")
    fig.tight_layout()
    fig.savefig(out / "fig_heldout_validation.png", dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = dec.load_model(MODEL_PKL)
    if bundle.get("glm_model") is None:
        print("NOTE: the pickle has no 1-state GLM (re-run run_glmhmm.py with the "
              "patch); comparison vs GLM will be skipped.")

    tables, meta = load_sessions()
    meta.to_csv(OUT_DIR / "sessions.csv", index=False)
    print("\n=== Held-out sessions ===")
    print(meta.groupby(["cohort", "status"]).size().to_string())
    print(f"{meta.loc[meta.status == 'ok', 'animal'].nunique()} animals pass the training criteria")

    print("\n=== (C) Comparability with the training dataset (session medians) ===")
    print(training_comparison(meta[meta.status == "ok"], tables).to_string())

    ses, ani = part_a(bundle, tables)
    ses.to_csv(OUT_DIR / "loglik_per_session.csv", index=False)
    ani.to_csv(OUT_DIR / "loglik_per_animal.csv", index=False)
    print("\n=== (A) Held-out log-likelihood per trial (animal means) ===")
    print(ani.round(4).to_string(index=False))
    if "ll_glm" in ani:
        t = paired(ani["ll_glmhmm"], ani["ll_glm"])
        print(f"GLM-HMM vs GLM: median gain {t['median_diff']:+.4f} nats/trial, "
              f"{t['frac_pos']*100:.0f}% of animals > 0, Wilcoxon p={t['p']:.3g} (n={t['n']})")
    t = paired(ani["ll_glmhmm"], ani["ll_chance"])
    print(f"GLM-HMM vs chance: median gain {t['median_diff']:+.4f} nats/trial, p={t['p']:.3g}")

    df = decode_all(bundle, tables)
    df.to_csv(OUT_DIR / "decoded_states.csv", index=False)
    print(f"\nExploit occupancy (median across animals): "
          f"{df.groupby('animal')['is_exploit'].mean().median():.3f}")

    res = part_b(df)
    names = {"V1": "P(better): exploit - explore (L_Random window)",
             "V2": "P(explore): trials 6-10 after switch - trials 0-3",
             "V3": "median RT: explore - exploit"}
    print("\n=== (B) Behavioral validation on new animals ===")
    for k, (tab, t) in res.items():
        tab.to_csv(OUT_DIR / f"{k}_per_animal.csv")
        print(f"{k} {names[k]}: median diff {t['median_diff']:+.3f}, "
              f"{t['frac_pos']*100:.0f}% of animals > 0, Wilcoxon p={t['p']:.3g} (n={t['n']})")
    figure(ani, res, OUT_DIR)
    print(f"\nResults in {OUT_DIR}")


if __name__ == "__main__":
    main()
