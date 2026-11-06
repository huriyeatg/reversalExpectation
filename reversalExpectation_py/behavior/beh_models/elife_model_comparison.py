"""
elife_model_comparison.py
=========================
Phase 2, step 2: cross-validated comparison of the eLife model family, the
variable-hazard belief models and reviewer 3's non-anticipating choice-history
agent, on the 31 naive mice.

Models (elife_models.py):
    WSLS, Q_RPE, DFQ_RPE, FQ_RPE, FQ_RPE_CK, DFQ_RPE_CK          eLife, as in MATLAB
    belief_exact, belief_CK_exact     eLife belief models with the exact Bayes update
    belief, belief_CK                 eLife belief models as in MATLAB (literal update)
    belief_vhr, belief_CK_vhr         variable hazard H(tau) = sigmoid(a + b * tau)
    R3_agent                          repeat / win-stay / lose-shift / random

Data: the cohort of every other analysis (naive mice, sessions meeting
criteria, >= 20 responded trials: 594 sessions), with the same 5 session folds
stratified by animal (hmmGlm._session_folds, seed 0).

Two schemes, both fitted PER ANIMAL:
    reset  : every session starts from the initial latent state; the model is
             fitted on the animal's training sessions and scored on its test
             sessions.                                       (primary analysis)
    concat : the eLife layout (merge_sessions.m): sessions in date order joined
             by 20 missed trials, latents carried across sessions. The model is
             fitted on the concatenation of the training sessions, and scored on
             the test sessions inside the concatenation of ALL the animal's
             sessions (so the latent state at a test session reflects the
             preceding history, as it would for the animal).

Held-out score: log-likelihood per trial on a COMMON set of trials for every
model: responded trials that are not the first of a session and whose previous
trial was responded (the belief models of the eLife code exclude the others).

Secondary score: the same, restricted to L_Random-phase trials (tau >= 1),
where an anticipatory hazard should matter most. Animals with a single
session cannot be cross-validated by session and only enter the full fits.

Also, per animal and scheme, a full-data fit of every model (parameters,
negloglike, BIC as in fit_fun.m).

Resumable: results are appended to checkpoint files as they are produced.

Inputs  (relative to reversalExpectation_py/):
    analysis/bandit_R71_lesion.csv
Outputs (analysis/elife_model_comparison/):
    cv_checkpoint.csv      scheme, model, animal, fold, ll, n_trials
    full_fits.csv          scheme, model, animal, parameters, negloglike, bic
    cv_per_animal.csv      held-out LL per trial, per scheme x model x animal
    cv_summary.csv         per scheme x model: mean, SEM, rank, vs best model
    report.txt

Usage (from reversalExpectation_py/):
    python behavior/beh_models/elife_model_comparison.py --jobs 2
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import elife_models as em                    # noqa: E402

PY_ROOT = HERE.parents[1]
CSV_PATH = PY_ROOT / "analysis" / "bandit_R71_lesion.csv"
OUT_DIR = PY_ROOT / "analysis" / "elife_model_comparison"

MODEL_LIST = ["WSLS", "Q_RPE", "DFQ_RPE", "FQ_RPE", "FQ_RPE_CK", "DFQ_RPE_CK",
              "belief_exact", "belief_CK_exact", "belief", "belief_CK",
              "belief_vhr", "belief_CK_vhr", "R3_agent"]
SCHEMES = ["reset", "concat"]
N_FOLDS, FOLD_SEED, MIN_TRIALS = 5, 0, 20
N_STARTS = 8


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def session_tau(d: pd.DataFrame) -> np.ndarray:
    """Trials since criterion (belief_vhr.prepare_vhr): 0 in the criterion phase,
    0, 1, 2, ... in the L_Random phase, 0 for blocks without criterion."""
    t_block = d.groupby("block_idx", sort=False).cumcount().to_numpy(float)
    ttc = d["block_trial_to_crit"].to_numpy(float)
    tau = np.where(np.isnan(ttc), 0.0, t_block - ttc)
    return np.clip(tau, 0.0, None)


def load_cohort(csv_path: Path):
    """Same selection and session order as glmhmm_regularization.load_sessions."""
    cols = ["animal", "session_file", "date_number", "trial_idx", "choice", "rewarded",
            "block_idx", "block_trial_to_crit", "lesioned", "meets_criteria"]
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    df = df[df["lesioned"].isna() & (df["meets_criteria"] == True)]   # noqa: E712
    sessions = []
    for (animal, ses), d in df.groupby(["animal", "session_file"], sort=False):
        d = d.sort_values("trial_idx")
        if d["choice"].notna().sum() < MIN_TRIALS:
            continue
        sessions.append({"animal": str(animal), "session": ses,
                         "date": float(d["date_number"].iloc[0]),
                         "c": d["choice"].to_numpy(float), "r": d["rewarded"].to_numpy(float),
                         "tau": session_tau(d)})
    return sessions


def session_folds(animals, n_folds, seed):
    """Copy of hmmGlm._session_folds: CV folds stratified by animal."""
    animals = np.asarray(animals, dtype=object)
    rng = np.random.default_rng(seed)
    fold_assign = np.full(len(animals), -1, dtype=int)
    for a in pd.unique(animals):
        idx = rng.permutation(np.where(animals == a)[0])
        for f, chunk in enumerate(np.array_split(idx, n_folds)):
            fold_assign[chunk] = f
    return fold_assign


def layout(sess, scheme):
    """Concatenate sessions for a scheme. Returns c, r, tau, starts, session_id
    (index into `sess` per trial, -1 for gap trials)."""
    parts = [(s["c"], s["r"]) for s in sess]
    if scheme == "concat":
        c, r, starts = em.build_concat(parts)
    else:
        c, r, starts = em.build_reset(parts)
    tau = np.zeros(len(c))
    sid = np.full(len(c), -1)
    for i, (s, st) in enumerate(zip(sess, starts)):
        tau[st:st + len(s["c"])] = s["tau"]
        sid[st:st + len(s["c"])] = i
    return c, r, tau, starts, sid


def common_mask(c, starts):
    """Responded, not first of session, previous trial responded."""
    m = ~np.isnan(c)
    prev = np.r_[False, ~np.isnan(c[:-1])]
    m &= prev
    m[np.asarray(starts)] = False
    return m


# ---------------------------------------------------------------------------
# Nested starting points: a richer model is also started from the optimum of
# the model it nests, so its in-sample fit can never be worse than the nested
# one because of the optimiser (b = 0 / lambda = alpha reproduce it exactly).
# ---------------------------------------------------------------------------
NESTS = {"belief_vhr": "belief_exact", "belief_CK_vhr": "belief_CK_exact",
         "DFQ_RPE": "FQ_RPE", "DFQ_RPE_CK": "FQ_RPE_CK"}


def nested_starts(name, base):
    """Starting points for `name` built from the fitted parameters of the
    model it nests (base), or None."""
    if base is None:
        return None
    if name in ("belief_vhr", "belief_CK_vhr"):
        H = float(np.clip(base[0], 1e-5, 1 - 1e-5))
        a = np.log(H / (1 - H))
        rest = list(base[1:])                      # beta (+ alpha_ck, beta_ck)
        return [[a, 0.0] + rest, [a - 0.5, 0.05] + rest, [a - 1.5, 0.15] + rest]
    if name in ("DFQ_RPE", "DFQ_RPE_CK"):
        alpha = base[0]
        return [[alpha, alpha] + list(base[1:])]
    return None


def fit_order(models):
    """Fit nested (simpler) models before the models that nest them."""
    first = [m for m in models if m in NESTS.values()]
    return first + [m for m in models if m not in first]


# ---------------------------------------------------------------------------
# Per-animal work
# ---------------------------------------------------------------------------
def run_animal(args):
    animal, sess, folds, models, schemes, n_starts = args
    order = np.argsort([s["date"] for s in sess], kind="stable")
    sess = [sess[i] for i in order]
    folds = np.asarray(folds)[order]
    reset = {"reset": True, "concat": False}
    cv_rows, fit_rows = [], []
    for scheme in schemes:
        c_all, r_all, tau_all, st_all, sid_all = layout(sess, scheme)
        cm_all = common_mask(c_all, st_all)
        best_full, best_fold = {}, {}
        for name in fit_order(models):
            m = em.MODELS[name]
            # ---- full-data fit
            o = em.fit(name, c_all, r_all, st_all, reset[scheme], literal=True,
                       n_starts=n_starts, seed=0, tau=tau_all,
                       extra_x0=nested_starts(name, best_full.get(NESTS.get(name))))
            best_full[name] = o["params"]
            fit_rows.append({"scheme": scheme, "model": name, "animal": animal,
                             "n_sessions": len(sess), "negloglike": o["negloglike"],
                             "bic": o["bic"], "n_params": m.n_params,
                             "n_responded": int(np.sum(~np.isnan(c_all))),
                             "starts_agree": o["starts_agree"], "n_starts": o["n_starts"],
                             **{f"par_{l}": v for l, v in zip(m.labels, o["params"])}})
            # ---- cross-validation
            for f in range(N_FOLDS):
                tr = [s for s, k in zip(sess, folds) if k != f]
                te_idx = [i for i, k in enumerate(folds) if k == f]
                if not tr or not te_idx:
                    continue
                c_tr, r_tr, tau_tr, st_tr, _ = layout(tr, scheme)
                o = em.fit(name, c_tr, r_tr, st_tr, reset[scheme], literal=True,
                           n_starts=n_starts, seed=1 + f, tau=tau_tr,
                           extra_x0=nested_starts(name, best_fold.get((NESTS.get(name), f))))
                best_fold[(name, f)] = o["params"]
                if scheme == "reset":
                    te = [sess[i] for i in te_idx]
                    c_te, r_te, tau_te, st_te, _ = layout(te, scheme)
                    ll = em.trial_loglik(name, o["params"], c_te, r_te, st_te, True,
                                         True, tau_te)
                    msk = common_mask(c_te, st_te)
                    tau_sc = tau_te
                else:
                    ll = em.trial_loglik(name, o["params"], c_all, r_all, st_all, False,
                                         True, tau_all)
                    msk = cm_all & np.isin(sid_all, te_idx)
                    tau_sc = tau_all
                # secondary score: L_Random-phase trials only (tau >= 1), where an
                # anticipatory hazard should matter
                msk_lr = msk & (tau_sc >= 1)
                cv_rows.append({"scheme": scheme, "model": name, "animal": animal,
                                "fold": f, "ll": float(np.nansum(ll[msk])),
                                "n_trials": int(msk.sum()),
                                "n_nan": int(np.isnan(ll[msk]).sum()),
                                "ll_lrandom": float(np.nansum(ll[msk_lr])),
                                "n_lrandom": int(msk_lr.sum())})
    return animal, cv_rows, fit_rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarise(cv: pd.DataFrame, fits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    pa = (cv.groupby(["scheme", "model", "animal"])[["ll", "n_trials"]].sum()
            .assign(ll_per_trial=lambda x: x["ll"] / x["n_trials"]).reset_index())
    rows, lines = [], []
    for scheme, g in pa.groupby("scheme"):
        piv = g.pivot(index="animal", columns="model", values="ll_per_trial")
        mean = piv.mean().sort_values(ascending=False)
        best = mean.index[0]
        bic = (fits[fits["scheme"] == scheme].groupby("model")["bic"].sum())
        for rank, mdl in enumerate(mean.index, 1):
            d = piv[mdl] - piv[best]
            rows.append({"scheme": scheme, "model": mdl, "rank": rank,
                         "heldout_ll_per_trial": piv[mdl].mean(),
                         "sem_across_animals": piv[mdl].std(ddof=1) / np.sqrt(len(piv)),
                         "diff_vs_best": d.mean(),
                         "animals_worse_than_best": f"{int((d < 0).sum())}/{len(d)}"
                                                    if mdl != best else "",
                         "wilcoxon_p_vs_best": wilcoxon(d).pvalue if mdl != best else np.nan,
                         "bic_sum_full_fit": bic.get(mdl, np.nan),
                         "animals_best": int((piv.idxmax(axis=1) == mdl).sum())})
    summ = pd.DataFrame(rows)
    fmt = lambda x: x.to_string(index=False, float_format=lambda v: f"{v:.5g}")  # noqa: E731
    for scheme, g in summ.groupby("scheme", sort=False):
        lines += [f"--- scheme: {scheme}", fmt(g.drop(columns="scheme")), ""]
        # the key contrasts
        piv = pa[pa["scheme"] == scheme].pivot(index="animal", columns="model",
                                                values="ll_per_trial")
        for a, b in [("belief_vhr", "belief_exact"), ("belief_CK_vhr", "belief_CK_exact"),
                     ("belief_CK_vhr", "R3_agent"), ("belief_CK_vhr", "FQ_RPE_CK"),
                     ("belief_CK_vhr", "DFQ_RPE_CK"), ("belief_CK_exact", "belief_CK")]:
            if a in piv and b in piv:
                d = piv[a] - piv[b]
                lines.append(f"  {a} - {b}: {d.mean():+.5f} nats/trial "
                             f"(median {d.median():+.5f}), {int((d > 0).sum())}/{len(d)} mice "
                             f"better, Wilcoxon p = {wilcoxon(d).pvalue:.2g}")
        lines.append("")
        # same contrasts on L_Random-phase trials only
        plr = (cv[cv["scheme"] == scheme].groupby(["model", "animal"])[["ll_lrandom", "n_lrandom"]]
               .sum().assign(v=lambda x: x["ll_lrandom"] / x["n_lrandom"])["v"].unstack("model"))
        lines.append("  L_Random-phase trials only (tau >= 1):")
        for a, b in [("belief_vhr", "belief_exact"), ("belief_CK_vhr", "belief_CK_exact"),
                     ("belief_CK_vhr", "R3_agent"), ("belief_CK_vhr", "DFQ_RPE_CK")]:
            if a in plr and b in plr:
                d = plr[a] - plr[b]
                lines.append(f"  {a} - {b}: {d.mean():+.5f} nats/trial "
                             f"(median {d.median():+.5f}), {int((d > 0).sum())}/{len(d)} mice "
                             f"better, Wilcoxon p = {wilcoxon(d).pvalue:.2g}")
        lines.append("")
    return pa, summ, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--models", nargs="+", default=MODEL_LIST)
    ap.add_argument("--schemes", nargs="+", default=SCHEMES)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--n-starts", type=int, default=N_STARTS)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    sessions = load_cohort(args.csv)
    animals = [s["animal"] for s in sessions]
    folds = session_folds(animals, N_FOLDS, FOLD_SEED)
    print(f"[data] {len(sessions)} sessions, {len(set(animals))} animals, "
          f"{sum(int(np.sum(~np.isnan(s['c']))) for s in sessions)} responded trials", flush=True)

    ck_cv, ck_fit = out / "cv_checkpoint.csv", out / "full_fits.csv"
    done = set(pd.read_csv(ck_fit, dtype={"animal": str})["animal"]) if ck_fit.exists() else set()
    todo = [a for a in dict.fromkeys(animals)
            if a not in done and (args.animals is None or a in args.animals)]
    jobs = []
    for a in todo:
        idx = [i for i, x in enumerate(animals) if x == a]
        jobs.append((a, [sessions[i] for i in idx], folds[idx], args.models, args.schemes,
                     args.n_starts))

    t0 = time.time()
    with Pool(args.jobs) as pool:
        for k, (a, cv_rows, fit_rows) in enumerate(pool.imap_unordered(run_animal, jobs), 1):
            pd.DataFrame(cv_rows).to_csv(ck_cv, mode="a", header=not ck_cv.exists(), index=False)
            pd.DataFrame(fit_rows).to_csv(ck_fit, mode="a", header=not ck_fit.exists(),
                                          index=False)
            print(f"[{k}/{len(jobs)}] animal {a} done ({time.time() - t0:.0f}s)", flush=True)

    cv = pd.read_csv(ck_cv, dtype={"animal": str})
    fits = pd.read_csv(ck_fit, dtype={"animal": str})
    pa, summ, txt = summarise(cv, fits)
    pa.to_csv(out / "cv_per_animal.csv", index=False)
    summ.to_csv(out / "cv_summary.csv", index=False)
    head = (f"Model comparison, 5-fold CV by session, per-animal fits, "
            f"{pa['animal'].nunique()} mice\nheld-out LL per trial on a common trial set "
            f"(responded, not first of session, previous trial responded)\n"
            f"{'=' * 78}\n")
    (out / "report.txt").write_text(head + txt, encoding="utf-8")
    print(head + txt)


if __name__ == "__main__":
    main()
