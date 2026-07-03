from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Shared clinical variable generators
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def _clin_age(n, mean=56, sd=14, lo=20, hi=90):
    return rng.normal(mean, sd, n).clip(lo, hi).astype(int)

def _clin_bmi(n, mean=25.5, sd=4.5, lo=15, hi=45):
    return np.round(rng.normal(mean, sd, n).clip(lo, hi), 1)

def _clin_sbp(n, mean=132, sd=17, lo=85, hi=200):
    return np.round(rng.normal(mean, sd, n).clip(lo, hi), 0).astype(int)

def _clin_glucose(n, mean=5.8, sd=1.6, lo=3.0, hi=16.0):
    return np.round(rng.normal(mean, sd, n).clip(lo, hi), 2)

def _clin_chol(n, mean=5.1, sd=1.1, lo=2.5, hi=9.5):
    return np.round(rng.normal(mean, sd, n).clip(lo, hi), 2)

def _clin_ids(n, prefix="P"):
    return [f"{prefix}{str(i).zfill(5)}" for i in range(1, n + 1)]


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Advanced Statistics Examples
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def make_gee_example() -> pd.DataFrame:
    """Longitudinal RCT: 5 timepoints, 3 arms, rich covariates, within-subject correlation."""
    n = 200
    records = []
    for i in range(1, n + 1):
        arm = rng.choice(["Active High", "Active Low", "Placebo"])
        site = rng.choice(["Site A", "Site B", "Site C", "Site D", "Site E"])
        age = rng.integers(30, 82)
        sex = rng.choice(["Male", "Female"])
        bmi = _clin_bmi(1)[0]
        base_sbp = rng.normal(148, 13)
        base_dbp = rng.normal(88, 9)
        base_hr = rng.normal(76, 10)
        subj_effect = rng.normal(0, 4)
        for t_idx, week in enumerate([0, 2, 4, 8, 12]):
            trt_map = {"Active High": -4.5, "Active Low": -2.8, "Placebo": -0.8}
            trend = trt_map[arm]
            noise = rng.normal(0, 5)
            records.append({
                "subject_id": f"S{str(i).zfill(5)}",
                "site": site, "arm": arm,
                "week": week, "time": t_idx,
                "sbp": round(base_sbp + trend * t_idx + subj_effect + noise, 1),
                "dbp": round(base_dbp + trend * 0.55 * t_idx + subj_effect * 0.6 + rng.normal(0, 3.5), 1),
                "heart_rate": round(base_hr - 0.3 * t_idx + rng.normal(0, 4), 1),
                "adverse_events": max(0, rng.poisson(0.4 + 0.1 * t_idx) - (0 if arm == "Placebo" else 0)),
                "age": age, "sex": sex, "bmi": bmi,
            })
    return pd.DataFrame(records)


def make_propensity_score_example() -> pd.DataFrame:
    """Observational study: rich confounders, treatment effect heterogeneity, multiple outcomes."""
    n = 800
    age = _clin_age(n, 60, 13)
    sex = rng.choice(["Male", "Female"], n)
    bmi = _clin_bmi(n)
    sbp = _clin_sbp(n)
    dbp = np.round(sbp * 0.5 + rng.normal(30, 6, n), 0).astype(int)
    glucose = _clin_glucose(n)
    cholesterol = _clin_chol(n)
    hdl = np.round(rng.normal(1.25, 0.35, n).clip(0.45, 2.9), 2)
    triglycerides = np.round(rng.lognormal(0.35, 0.5, n).clip(0.4, 9.0), 2)
    egfr = np.round(rng.normal(78, 18, n).clip(18, 130), 1)
    albumin = np.round(rng.normal(40, 5, n).clip(22, 52), 1)
    smoking = rng.choice(["Never", "Former", "Current"], n, p=[0.42, 0.32, 0.26])
    alcohol = rng.choice(["None", "Moderate", "Heavy"], n, p=[0.50, 0.35, 0.15])
    education = rng.choice(["Primary", "Secondary", "Tertiary"], n, p=[0.28, 0.45, 0.27])
    income = rng.choice(["Low", "Medium", "High"], n, p=[0.30, 0.48, 0.22])
    exercise = rng.choice(["None", "1-2x/wk", "3+x/wk"], n, p=[0.40, 0.35, 0.25])
    comorbidity = rng.poisson(1.8, n).clip(0, 7)
    prev_hosp = rng.poisson(0.6, n).clip(0, 5)

    # Realistic propensity score with non-linear terms
    ps_linear = (-2.8 + 0.035 * age + 0.5 * (sex == "Male") + 0.06 * bmi
                 + 0.012 * sbp + 0.22 * glucose + 0.08 * cholesterol
                 + 0.55 * (smoking == "Current") + 0.30 * (smoking == "Former")
                 + 0.40 * (alcohol == "Heavy") + 0.15 * (alcohol == "Moderate")
                 - 0.35 * (education == "Tertiary") - 0.20 * (income == "High")
                 - 0.25 * (exercise == "3+x/wk") + 0.30 * comorbidity
                 - 0.010 * egfr + 0.08 * (prev_hosp > 0)
                 + rng.normal(0, 0.6, n))
    ps = 1 / (1 + np.exp(-ps_linear))
    treatment = (rng.random(n) < ps).astype(int)

    # Multiple outcomes with treatment effect
    y_continuous = np.round(
        4.0 + 1.2 * treatment + 0.03 * age - 0.40 * (smoking == "Current")
        - 0.15 * comorbidity + 0.008 * egfr + rng.normal(0, 1.0, n), 2
    )
    y_binary_logit = (-1.5 + 0.60 * treatment + 0.03 * age + 0.15 * (sbp > 140)
                      + 0.40 * (smoking == "Current") + rng.normal(0, 0.3, n))
    y_binary = (rng.random(n) < 1 / (1 + np.exp(-y_binary_logit))).astype(int)

    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age, "sex": sex, "bmi": bmi, "sbp": sbp, "dbp": dbp,
        "glucose": glucose, "cholesterol": cholesterol,
        "hdl": hdl, "triglycerides": triglycerides,
        "egfr": egfr, "albumin": albumin,
        "smoking": smoking, "alcohol": alcohol,
        "education": education, "income": income,
        "exercise_freq": exercise,
        "comorbidity_count": comorbidity,
        "prev_hospitalizations": prev_hosp,
        "treatment": treatment,
        "treatment_group": np.where(treatment == 1, "Treated", "Control"),
        "outcome_continuous": y_continuous,
        "outcome_binary": y_binary,
    })


def make_sensitivity_analysis_example() -> pd.DataFrame:
    """RCT with complex missing data: MNAR, MAR, and MCAR patterns."""
    n = 500
    arm = rng.choice(["Experimental", "Standard", "Combination"], n)
    age = _clin_age(n, 56, 14)
    sex = rng.choice(["Male", "Female"], n)
    bmi = _clin_bmi(n)
    site = rng.choice(["Center 1", "Center 2", "Center 3", "Center 4", "Center 5"], n)
    bsl_1 = np.round(rng.normal(32, 6, n).clip(10, 50), 1)
    bsl_2 = np.round(rng.normal(28, 5, n).clip(8, 42), 1)
    bsl_3 = np.round(rng.normal(22, 4.5, n).clip(5, 38), 1)
    adherence = rng.choice(["Full", "Partial", "Poor"], n, p=[0.55, 0.30, 0.15])

    trt_eff = {"Experimental": -4.5, "Standard": -2.0, "Combination": -6.2}
    fup_1 = np.round(bsl_1 + np.array([trt_eff[a] for a in arm]) * rng.uniform(0.7, 1.3, n)
                    + rng.normal(0, 4.5, n), 1)
    fup_2 = np.round(bsl_2 + np.array([trt_eff[a] for a in arm]) * 0.7
                    + rng.normal(0, 3.8, n), 1)
    fup_3 = np.round(bsl_3 + np.array([trt_eff[a] for a in arm]) * 0.5
                    + rng.normal(0, 3.5, n), 1)

    # Complex missing data mechanisms
    # MNAR: sicker patients more likely to have missing followup
    p_miss_1 = 1 / (1 + np.exp(-(-2.5 + 0.05 * (bsl_1 - 32) + 0.6 * (adherence == "Poor"))))
    p_miss_2 = 1 / (1 + np.exp(-(-2.2 + 0.04 * (bsl_2 - 28) + 0.8 * (adherence == "Poor"))))
    p_miss_3 = 1 / (1 + np.exp(-(-2.0 + 0.03 * (bsl_3 - 22) + 0.5 * (adherence == "Poor")
                                   - 0.3 * (arm == "Combination"))))

    miss_1 = rng.random(n) < p_miss_1 * 0.5
    miss_2 = rng.random(n) < p_miss_2 * 0.5
    miss_3 = rng.random(n) < p_miss_3 * 0.5
    fup_1[miss_1] = np.nan
    fup_2[miss_2] = np.nan
    fup_3[miss_3] = np.nan

    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "arm": arm, "site": site,
        "age": age, "sex": sex, "bmi": bmi,
        "adherence": adherence,
        "baseline_score1": bsl_1, "baseline_score2": bsl_2, "baseline_score3": bsl_3,
        "followup_score1": fup_1, "followup_score2": fup_2, "followup_score3": fup_3,
        "change_score1": np.round(fup_1 - bsl_1, 1),
        "change_score2": np.round(fup_2 - bsl_2, 1) if not all(np.isnan(fup_2)) else np.nan,
    })


def make_counterfactual_example() -> pd.DataFrame:
    """Observational data with rich DAG structure for causal inference."""
    n = 600
    age = _clin_age(n, 58, 13)
    sex = rng.choice(["Male", "Female"], n)
    bmi = _clin_bmi(n)
    education = rng.choice(["Low", "Medium", "High"], n, p=[0.30, 0.45, 0.25])
    income = rng.choice(["Low", "Medium", "High"], n, p=[0.28, 0.50, 0.22])
    bsl_severity = np.round(rng.normal(7.0, 2.2, n).clip(1, 10), 1)
    smoking = rng.choice(["Never", "Former", "Current"], n, p=[0.40, 0.32, 0.28])
    alcohol = rng.choice([0, 1], n, p=[0.52, 0.48])
    exercise = rng.choice([0, 1, 2], n, p=[0.35, 0.40, 0.25])
    comorbidity = rng.poisson(1.6, n).clip(0, 6)
    prev_meds = rng.choice([0, 1, 2, 3], n, p=[0.25, 0.30, 0.28, 0.17])

    # Treatment assignment (multiple causes)
    p_treat = 1 / (1 + np.exp(-(-1.8 + 0.18 * bsl_severity + 0.55 * (smoking == "Current")
                                 + 0.30 * comorbidity - 0.25 * (exercise == 2)
                                 - 0.15 * (education == "High") + rng.normal(0, 0.35, n))))
    treatment = (rng.random(n) < p_treat).astype(int)

    # Potential outcomes with heterogeneous treatment effect
    y0 = (5.0 + 0.55 * bsl_severity - 0.35 * (smoking == "Former")
          - 0.65 * (smoking == "Current") - 0.20 * comorbidity
          + 0.15 * (exercise == 2) + 0.012 * age + rng.normal(0, 1.6, n))
    cate = 3.0 + 0.4 * bsl_severity - 0.3 * (comorbidity > 2) + rng.normal(0, 0.5, n)
    y1 = y0 + np.clip(cate, 0.5, None)
    outcome = np.where(treatment == 1, y1, y0)

    # Secondary outcome (binary)
    y0_bin_logit = (-0.5 + 0.3 * bsl_severity - 0.15 * (exercise > 0) + rng.normal(0, 0.2, n))
    y0_bin = (rng.random(n) < 1 / (1 + np.exp(-y0_bin_logit))).astype(int)
    ate_bin = 0.25
    y1_bin_logit = y0_bin_logit + ate_bin + 0.1 * bsl_severity + rng.normal(0, 0.15, n)
    y1_bin = (rng.random(n) < 1 / (1 + np.exp(-y1_bin_logit))).astype(int)
    outcome_bin = np.where(treatment == 1, y1_bin, y0_bin)

    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age, "sex": sex, "bmi": bmi,
        "education": education, "income": income,
        "baseline_severity": bsl_severity,
        "smoking": smoking, "alcohol": alcohol,
        "exercise_level": exercise,
        "comorbidity_count": comorbidity,
        "previous_medications": prev_meds,
        "treatment": treatment,
        "treatment_group": np.where(treatment == 1, "Treated", "Control"),
        "outcome_continuous": np.round(outcome, 2),
        "outcome_binary": outcome_bin,
    })


def make_survival_advanced_example() -> pd.DataFrame:
    """Complex oncology dataset: staging, biomarkers, competing risks, treatment interactions."""
    n = 600
    age = _clin_age(n, 61, 12)
    sex = rng.choice(["Male", "Female"], n)
    stage = rng.choice(["I", "II", "III", "IV"], n, p=[0.16, 0.26, 0.35, 0.23])
    grade = rng.choice(["G1", "G2", "G3", "G4"], n, p=[0.12, 0.30, 0.38, 0.20])
    histology = rng.choice(["Adenocarcinoma", "Squamous", "Small Cell", "Large Cell"], n,
                           p=[0.40, 0.28, 0.17, 0.15])
    treatment = rng.choice(["Surgery Only", "Surgery+Chemo", "Surgery+Chemo+Radio",
                            "Chemo+Radio", "Palliative"], n,
                           p=[0.18, 0.28, 0.22, 0.20, 0.12])
    bmi = _clin_bmi(n)
    albumin = np.round(rng.normal(37, 6.5, n).clip(16, 52), 1)
    ldh = np.round(rng.lognormal(5.2, 0.38, n).clip(40, 900), 0).astype(int)
    hemoglobin = np.round(rng.normal(13.0, 1.8, n).clip(7, 18), 1)
    wbc = np.round(rng.lognormal(1.9, 0.3, n).clip(2, 20), 1)
    platelets = np.round(rng.normal(250, 80, n).clip(50, 600), 0).astype(int)
    nlr = np.round(wbc * 1000 / (platelets * rng.uniform(0.008, 0.015, n)), 2)
    ecog = rng.choice([0, 1, 2, 3], n, p=[0.40, 0.30, 0.20, 0.10])
    comorbidity = rng.poisson(1.4, n).clip(0, 5)
    tumor_size = np.round(rng.lognormal(3.0, 0.6, n).clip(0.5, 15), 1)

    risk = (0.035 * age + 0.7 * (stage == "III") + 1.5 * (stage == "IV")
            + 0.3 * (grade == "G3") + 0.6 * (grade == "G4")
            - 0.35 * (treatment == "Surgery+Chemo")
            - 0.5 * (treatment == "Surgery+Chemo+Radio")
            + 0.4 * (treatment == "Palliative")
            - 0.025 * albumin + 0.4 * (ecog >= 2) + 0.15 * comorbidity
            + 0.08 * tumor_size + 0.3 * (nlr > 4)
            + rng.normal(0, 0.4, n))

    base_hazard = np.exp(risk - 2.0)
    time_os = np.round(rng.exponential(1 / np.clip(base_hazard, 0.002, 0.5,), n)).clip(1, 90).astype(int)
    time_pfs = np.round(time_os * rng.beta(5, 3, n)).clip(1, 90).astype(int)
    event_os = rng.choice([0, 1], n, p=[0.35, 0.65])
    event_pfs = rng.choice([0, 1], n, p=[0.28, 0.72])
    cause = rng.choice(["Cancer Death", "Cardiovascular", "Other", "Alive"], n,
                       p=[0.42, 0.10, 0.08, 0.40])

    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age, "sex": sex,
        "stage": stage, "grade": grade, "histology": histology,
        "treatment": treatment,
        "treatment_group": np.where(treatment == "Palliative", "Palliative", "Active"),
        "bmi": bmi, "albumin": albumin, "ldh": ldh,
        "hemoglobin": hemoglobin, "wbc": wbc, "platelets": platelets,
        "nlr": nlr, "ecog": ecog,
        "comorbidity_count": comorbidity,
        "tumor_size_cm": tumor_size,
        "risk_score": np.round(risk, 3),
        "time_os": time_os, "event_os": event_os,
        "time_pfs": time_pfs, "event_pfs": event_pfs,
        "cause_of_death": cause,
    })


def make_markov_model_example() -> pd.DataFrame:
    """Disease progression: 7-state Markov model with covariates."""
    n = 300
    states = ["Healthy", "At Risk", "Preclinical", "Mild Disease",
              "Moderate Disease", "Severe Disease", "Death"]
    n_states = len(states)
    data = []
    for i in range(1, n + 1):
        age = rng.integers(35, 85)
        sex = rng.choice(["Male", "Female"])
        bmi = _clin_bmi(1)[0]
        smoking = rng.choice([0, 1], p=[0.55, 0.45])
        start_idx = rng.choice(range(n_states - 1), p=[0.30, 0.22, 0.18, 0.15, 0.10, 0.05])
        current = start_idx
        for m in range(0, 73, 3):
            if current == n_states - 1:
                break
            # Transition depends on covariates
            age_eff = 0.3 * (age > 60)
            smoke_eff = 0.25 * smoking
            bmi_eff = 0.15 * (bmi > 30)
            progress_prob = min(0.35, 0.05 + 0.01 * current + age_eff + smoke_eff + bmi_eff)
            regress_prob = 0.03 if not smoking else 0.01
            stay_prob = max(0, 1 - progress_prob - regress_prob)
            total = stay_prob + progress_prob + regress_prob
            trans = rng.choice(3, p=[stay_prob/total, progress_prob/total, regress_prob/total])
            if trans == 1:
                current = min(current + 1, n_states - 1)
            elif trans == 2:
                current = max(current - 1, 0)
            data.append({
                "subject_id": f"S{str(i).zfill(5)}",
                "month": m,
                "state": states[current],
                "state_index": current,
                "age": age, "sex": sex,
                "bmi": bmi, "smoking": smoking,
            })
    return pd.DataFrame(data)


def make_bayesian_example() -> pd.DataFrame:
    """RCT data for Bayesian analysis: multiple endpoints, prior elicitation support."""
    n = 350
    arm = rng.choice(["New Biologic", "Standard Biologic", "Placebo"], n,
                     p=[0.40, 0.35, 0.25])
    age = _clin_age(n, 52, 15)
    sex = rng.choice(["Male", "Female"], n)
    bmi = _clin_bmi(n)
    bsl_das28 = np.round(rng.normal(5.8, 1.2, n).clip(2.0, 8.5), 1)
    bsl_haq = np.round(rng.normal(1.2, 0.5, n).clip(0, 2.8), 2)
    bsl_pain = np.round(rng.normal(62, 18, n).clip(10, 100), 0).astype(int)
    bsl_crp = np.round(rng.lognormal(1.2, 0.7, n).clip(0.2, 60), 2)
    disease_duration = rng.lognormal(1.8, 0.6, n).clip(0.2, 30)
    prior_mtx = rng.choice(["MTX Naive", "MTX IR", "TNF IR"], n, p=[0.30, 0.40, 0.30])
    steroid_use = rng.choice([0, 1], n, p=[0.45, 0.55])

    trt_eff = {"New Biologic": -2.2, "Standard Biologic": -1.4, "Placebo": -0.3}
    fup_das28 = np.round(bsl_das28 + np.array([trt_eff[a] for a in arm])
                         + rng.normal(0, 0.8, n), 1)
    fup_haq = np.round(bsl_haq + np.array([trt_eff[a] * 0.15 for a in arm])
                       + rng.normal(0, 0.2, n), 2)
    fup_pain = np.round(bsl_pain + np.array([trt_eff[a] * 8 for a in arm])
                        + rng.normal(0, 12, n), 0).astype(int)
    fup_crp = np.round(bsl_crp * np.exp(np.array([trt_eff[a] * 0.25 for a in arm]))
                       + rng.normal(0, 2, n), 2)

    responder_acr20 = ((bsl_das28 - fup_das28) > 1.2).astype(int)
    responder_acr50 = ((bsl_das28 - fup_das28) > 2.0).astype(int)
    ae_serious = rng.choice([0, 1], n, p=[0.88, 0.12])
    ae_any = rng.choice([0, 1], n, p=[0.55, 0.45])

    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "arm": arm, "age": age, "sex": sex, "bmi": bmi,
        "disease_duration_years": np.round(disease_duration, 1),
        "prior_medication": prior_mtx,
        "steroid_use": steroid_use,
        "baseline_das28": bsl_das28, "baseline_haq": bsl_haq,
        "baseline_pain_vas": bsl_pain, "baseline_crp": bsl_crp,
        "followup_das28": fup_das28, "followup_haq": fup_haq,
        "followup_pain_vas": fup_pain, "followup_crp": fup_crp,
        "das28_change": np.round(bsl_das28 - fup_das28, 2),
        "acr20_response": responder_acr20,
        "acr50_response": responder_acr50,
        "ae_serious": ae_serious, "ae_any": ae_any,
    })


def make_latin_square_example() -> pd.DataFrame:
    """Crossover trial: 4x4 Latin square with washout and carryover."""
    sequences = [
        "ABCD", "BADC", "CDAB", "DCBA",
        "ABDC", "BCAD", "CDBA", "DACB",
        "ACBD", "BDAC", "CABD", "DBCA",
    ]
    periods = [1, 2, 3, 4]
    formulations = {"A": "Drug A 50mg", "B": "Drug B 100mg",
                    "C": "Drug C 75mg", "D": "Placebo"}
    data = []
    subj_id = 1
    for seq in sequences:
        for rep in range(4):
            age = rng.integers(28, 72)
            sex = rng.choice(["Male", "Female"])
            bmi = _clin_bmi(1)[0]
            base_response = rng.normal(90, 10)
            carryover_effect = 0
            for p_idx, period in enumerate(periods):
                form_letter = seq[p_idx]
                effect = {"A": -3.0, "B": -6.5, "C": -10.0, "D": 0.5}[form_letter]
                period_effect = {1: 0, 2: 1.5, 3: 2.8, 4: 4.2}[period]
                val = base_response + effect + period_effect + carryover_effect + rng.normal(0, 3.5)
                data.append({
                    "subject_id": f"S{str(subj_id).zfill(5)}",
                    "sequence": seq, "period": period,
                    "formulation": formulations[form_letter],
                    "formulation_code": form_letter,
                    "response": round(val, 2),
                    "age": age, "sex": sex, "bmi": bmi,
                })
                carryover_effect = effect * 0.15
            subj_id += 1
    return pd.DataFrame(data)


def make_meta_analysis_example() -> pd.DataFrame:
    """Meta-analysis: 20 studies with moderators, risk of bias, publication bias signal."""
    studies = [
        ("Beijing Union Hospital", 185, 0.68, 0.09, "RCT", "Low", "China"),
        ("Shanghai Renji", 142, 0.73, 0.11, "RCT", "Low", "China"),
        ("Guangzhou Nanfang", 128, 0.55, 0.10, "RCT", "Unclear", "China"),
        ("Chengdu Huaxi", 105, 0.82, 0.13, "RCT", "Low", "China"),
        ("Wuhan Tongji", 98, 0.70, 0.12, "Observational", "High", "China"),
        ("Nanjing Zhongda", 112, 0.52, 0.08, "RCT", "Low", "China"),
        ("Hangzhou Sir Run Run", 88, 0.76, 0.12, "RCT", "Unclear", "China"),
        ("Shenyang Shengjing", 75, 0.61, 0.14, "Observational", "High", "China"),
        ("Tianjin Medical", 95, 0.79, 0.10, "RCT", "Low", "China"),
        ("Changsha Xiangya", 115, 0.58, 0.09, "RCT", "Low", "China"),
        ("Mayo Clinic, MN", 220, 0.64, 0.07, "RCT", "Low", "USA"),
        ("Mass General, MA", 195, 0.71, 0.08, "RCT", "Low", "USA"),
        ("Charit茅, Berlin", 168, 0.66, 0.09, "RCT", "Low", "Germany"),
        ("St Thomas', London", 155, 0.60, 0.10, "Observational", "Unclear", "UK"),
        ("Karolinska, Stockholm", 138, 0.75, 0.11, "RCT", "Low", "Sweden"),
        ("Toronto General", 178, 0.69, 0.08, "RCT", "Low", "Canada"),
        ("Royal Melbourne", 122, 0.57, 0.12, "Observational", "High", "Australia"),
        ("Seoul National", 108, 0.80, 0.13, "RCT", "Unclear", "Korea"),
        ("Singapore General", 92, 0.72, 0.14, "RCT", "Low", "Singapore"),
        ("Sao Paulo Clinicas", 85, 0.63, 0.15, "Observational", "High", "Brazil"),
    ]
    rows = []
    for (study, n, hr, se, design, bias, country) in studies:
        rows.append({
            "study": study, "sample_size": n,
            "effect_size": hr,
            "ci_lower": round(hr - 1.96 * se, 3),
            "ci_upper": round(hr + 1.96 * se, 3),
            "standard_error": se,
            "study_design": design,
            "risk_of_bias": bias,
            "country": country,
            "weight_random": round(1 / se**2, 3),
        })
    return pd.DataFrame(rows)


def make_mediation_example() -> pd.DataFrame:
    """Mediation: X鈫扢1鈫扢2鈫扽 with multiple mediators and confounders."""
    n = 450
    age = _clin_age(n, 52, 14)
    sex = rng.choice(["Male", "Female"], n)
    bmi = _clin_bmi(n)
    education = rng.choice(["Low", "Medium", "High"], n, p=[0.25, 0.48, 0.27])
    ses = rng.normal(0, 1, n)  # Socioeconomic status (standardized)

    # X: chronic stress (exposure)
    x_stress = np.round(3.0 - 0.25 * ses + 0.15 * (education == "Low")
                        - 0.10 * (education == "High") + rng.normal(0, 1.2, n), 1)

    # M1: inflammatory markers (mediator 1)
    m1_crp = np.round(np.exp(0.2 + 0.35 * x_stress + 0.10 * bmi * 0.1
                             - 0.12 * ses + rng.normal(0, 0.35, n)), 2)
    m1_il6 = np.round(np.exp(0.5 + 0.30 * x_stress + 0.08 * bmi * 0.1
                             + rng.normal(0, 0.40, n)), 2)

    # M2: metabolic markers (mediator 2, influenced by M1)
    m2_hba1c = np.round(4.8 + 0.15 * m1_crp + 0.08 * x_stress + 0.03 * bmi
                        + rng.normal(0, 0.3, n), 1)
    m2_sbp = _clin_sbp(n)

    # Y: cardiovascular event risk (outcome)
    y_risk = np.round(
        100 + 2.8 * x_stress + 1.5 * m1_crp + 0.6 * m1_il6
        + 1.2 * m2_hba1c + 0.02 * m2_sbp + 0.5 * age * 0.1
        + rng.normal(0, 6, n), 1
    )

    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "stress_score": x_stress,
        "crp": m1_crp, "il6": m1_il6,
        "hba1c": m2_hba1c, "sbp": m2_sbp,
        "cvd_risk_score": y_risk,
        "age": age, "sex": sex, "bmi": bmi,
        "education": education, "ses_index": np.round(ses, 2),
    })


def make_mixed_effects_example() -> pd.DataFrame:
    """Multi-center longitudinal: 8 centers, rich random effects structure."""
    centers = ["Beijing", "Shanghai", "Guangzhou", "Chengdu",
               "Wuhan", "Nanjing", "Hangzhou", "Tianjin"]
    records = []
    subj = 1
    for center in centers:
        center_eff = rng.normal(0, 3.5)
        for _ in range(50):
            arm = rng.choice(["Drug", "Drug+Behavioral", "Placebo"])
            age = rng.integers(28, 80)
            sex = rng.choice(["Male", "Female"])
            bmi = _clin_bmi(1)[0]
            base_sbp = rng.normal(142, 12)
            base_dbp = rng.normal(88, 9)
            base_bmi_actual = _clin_bmi(1)[0]
            subj_slope = rng.normal(0, 1.8)
            for t_idx, week in enumerate([0, 4, 8, 12, 24, 36]):
                trt_map = {"Drug": -1.8, "Drug+Behavioral": -3.2, "Placebo": -0.3}
                trt_eff = trt_map[arm]
                sbp = base_sbp + trt_eff * t_idx + center_eff + subj_slope * t_idx + rng.normal(0, 4.5)
                dbp = base_dbp + trt_eff * 0.6 * t_idx + center_eff * 0.5 + subj_slope * 0.4 * t_idx + rng.normal(0, 3)
                records.append({
                    "subject_id": f"S{str(subj).zfill(5)}",
                    "center": center, "arm": arm,
                    "week": week, "time": t_idx,
                    "sbp": round(sbp, 1), "dbp": round(dbp, 1),
                    "age": age, "sex": sex, "bmi": base_bmi_actual,
                })
            subj += 1
    return pd.DataFrame(records)


def make_nhanes_analysis_example() -> pd.DataFrame:
    """Complex survey: NHANES-style with strata, PSU, weights, 25+ health variables."""
    n = 800
    strata = rng.choice(
        ["Urban-High SES", "Urban-Mid SES", "Urban-Low SES",
         "Suburban-High", "Suburban-Low",
         "Rural-High", "Rural-Low"], n,
        p=[0.15, 0.20, 0.15, 0.18, 0.12, 0.10, 0.10]
    )
    psu = rng.integers(1, 60, n)
    survey_weight = np.round(rng.lognormal(4.0, 0.45, n), 2)
    age = _clin_age(n, 45, 20, 18, 82)
    sex = rng.choice(["Male", "Female"], n)
    race = rng.choice(["Han", "Uyghur", "Tibetan", "Mongol", "Hui", "Other"], n,
                      p=[0.75, 0.05, 0.04, 0.04, 0.07, 0.05])
    bmi = _clin_bmi(n, 27.0, 5.8)
    waist = np.round(rng.normal(88, 14, n).clip(55, 140), 1)
    sbp = _clin_sbp(n, 130, 17)
    dbp = np.round(sbp * 0.52 + rng.normal(28, 6, n), 0).astype(int)
    glucose = _clin_glucose(n, 6.0, 2.0)
    hba1c = np.round(glucose * 0.68 + rng.normal(2.6, 0.35, n), 1)
    insulin = np.round(rng.lognormal(2.0, 0.5, n).clip(1, 50), 1)
    cholesterol = _clin_chol(n)
    hdl = np.round(rng.normal(1.22, 0.36, n).clip(0.4, 2.9), 2)
    ldl = np.round(cholesterol - hdl * 0.2 - rng.normal(0.3, 0.3, n), 2)
    triglycerides = np.round(rng.lognormal(0.35, 0.5, n).clip(0.3, 10), 2)
    creatinine = np.round(rng.normal(78, 25, n).clip(30, 250), 1)
    egfr = np.round(175 - 0.9 * age - 0.25 * creatinine + rng.normal(0, 10, n), 1)
    albumin = np.round(rng.normal(41, 4.5, n).clip(22, 53), 1)
    alt = np.round(rng.lognormal(3.2, 0.4, n).clip(5, 120), 0).astype(int)
    ast_val = np.round(rng.lognormal(3.0, 0.35, n).clip(5, 100), 0).astype(int)
    uric_acid = np.round(rng.normal(330, 80, n).clip(100, 600), 1)
    vitamin_d = np.round(rng.normal(55, 22, n).clip(8, 120), 1)
    hemoglobin = np.round(rng.normal(13.8, 1.6, n).clip(8, 18), 1)
    ferritin = np.round(rng.lognormal(4.5, 0.7, n).clip(5, 800), 1)
    smoking = rng.choice(["Never", "Former", "Current"], n, p=[0.48, 0.28, 0.24])
    alcohol = rng.choice(["None", "Light", "Moderate", "Heavy"], n, p=[0.45, 0.25, 0.20, 0.10])
    exercise = rng.choice(["Sedentary", "Light", "Moderate", "Vigorous"], n,
                          p=[0.30, 0.28, 0.27, 0.15])
    hypertension = (sbp >= 140).astype(int)
    diabetes = (hba1c >= 6.5).astype(int)
    metabolic_syndrome = ((waist > 90) & (hdl < 1.0) & (sbp > 130) & (glucose > 5.6)).astype(int)
    ckd = (egfr < 60).astype(int)

    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "strata": strata, "psu": psu, "survey_weight": survey_weight,
        "age": age, "sex": sex, "race": race,
        "bmi": bmi, "waist_cm": waist,
        "sbp": sbp, "dbp": dbp,
        "glucose": glucose, "hba1c": hba1c, "insulin": insulin,
        "cholesterol": cholesterol, "hdl": hdl, "ldl": ldl,
        "triglycerides": triglycerides,
        "creatinine": creatinine, "egfr": egfr, "albumin": albumin,
        "alt": alt, "ast": ast_val, "uric_acid": uric_acid,
        "vitamin_d": vitamin_d, "hemoglobin": hemoglobin, "ferritin": ferritin,
        "smoking": smoking, "alcohol": alcohol, "exercise": exercise,
        "hypertension": hypertension, "diabetes": diabetes,
        "metabolic_syndrome": metabolic_syndrome, "ckd": ckd,
    })


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Machine Learning Examples
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def make_ml_lr_example() -> pd.DataFrame:
    """Binary classification: 20 clinical predictors, non-linear relationships, interactions."""
    n = 800
    age = _clin_age(n, 59, 13)
    sex = rng.choice([0, 1], n)
    bmi = _clin_bmi(n)
    sbp = _clin_sbp(n)
    dbp = np.round(sbp * 0.5 + rng.normal(28, 7, n), 0).astype(int)
    glucose = _clin_glucose(n)
    hba1c = np.round(glucose * 0.65 + rng.normal(2.7, 0.35, n), 1)
    cholesterol = _clin_chol(n)
    hdl = np.round(rng.normal(1.22, 0.35, n).clip(0.4, 2.9), 2)
    ldl = np.round(cholesterol - hdl * 0.19 - rng.normal(0.25, 0.25, n), 2)
    triglycerides = np.round(rng.lognormal(0.35, 0.5, n).clip(0.3, 10), 2)
    creatinine = np.round(rng.normal(80, 26, n).clip(35, 260), 1)
    egfr = np.round(170 - 0.85 * age - 0.22 * creatinine + rng.normal(0, 9, n), 1)
    albumin = np.round(rng.normal(40, 5, n).clip(20, 53), 1)
    hemoglobin = np.round(rng.normal(13.5, 1.7, n).clip(7.5, 18.5), 1)
    wbc = np.round(rng.lognormal(2.0, 0.25, n).clip(2.5, 18), 1)
    platelets = np.round(rng.normal(240, 75, n).clip(60, 550), 0).astype(int)
    smoking = rng.choice([0, 1], n, p=[0.55, 0.45])
    alcohol = rng.choice([0, 1], n, p=[0.58, 0.42])
    family_hx = rng.choice([0, 1], n, p=[0.62, 0.38])
    prev_event = rng.choice([0, 1], n, p=[0.82, 0.18])

    # Complex risk model with interactions and non-linear terms
    lp = (-4.5
          + 0.04 * age
          + 0.40 * sex
          + 0.06 * bmi
          + 0.018 * sbp
          + 0.28 * glucose
          + 0.12 * cholesterol
          - 0.25 * np.log(hdl + 0.01)
          + 0.15 * np.log(triglycerides + 0.01)
          - 0.008 * egfr
          + 0.65 * smoking
          + 0.28 * alcohol
          + 0.85 * family_hx
          + 0.50 * prev_event
          + 0.15 * (age > 60) * smoking  # interaction
          + 0.10 * (bmi > 30) * (glucose > 7)  # interaction
          - 0.20 * (hdl > 1.5)  # non-linear protective
          + rng.normal(0, 0.45, n))
    prob = 1 / (1 + np.exp(-lp))
    outcome = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age, "sex": sex, "bmi": bmi,
        "sbp": sbp, "dbp": dbp,
        "glucose": glucose, "hba1c": hba1c,
        "cholesterol": cholesterol, "hdl": hdl, "ldl": ldl,
        "triglycerides": triglycerides,
        "creatinine": creatinine, "egfr": egfr, "albumin": albumin,
        "hemoglobin": hemoglobin, "wbc": wbc, "platelets": platelets,
        "smoking": smoking, "alcohol": alcohol,
        "family_history": family_hx, "previous_event": prev_event,
        "outcome": outcome,
    })


def make_ml_lasso_example() -> pd.DataFrame:
    """High-dimensional: 200 biomarkers, sparse true signal, correlated features."""
    n = 300
    p = 200
    # Generate correlated features via block structure
    n_blocks = 20
    block_size = p // n_blocks
    X = np.zeros((n, p))
    for b in range(n_blocks):
        base = rng.normal(0, 1, n)
        for j in range(block_size):
            col = b * block_size + j
            X[:, col] = base * 0.7 + rng.normal(0, 0.3, n)

    # Sparse true coefficients (15 non-zero out of 200)
    true_beta = np.zeros(p)
    signal_indices = [0, 13, 21, 35, 48, 56, 72, 84, 99, 108,
                      122, 137, 150, 168, 185]
    for idx in signal_indices:
        true_beta[idx] = rng.normal(0, 1.8, 1)[0]

    # Add non-linear interactions for a few features
    y_linear = X @ true_beta
    y_nonlinear = (0.3 * X[:, 0] * X[:, 13]   # interaction
                   + 0.2 * np.abs(X[:, 35])    # non-linear
                   + 0.15 * np.sin(X[:, 72])   # periodic
                   - 0.1 * np.exp(np.clip(X[:, 108], -2, 2)) * 0.5)
    y = y_linear + y_nonlinear + rng.normal(0, 2.0, n)

    cols = [f"biomarker_{i:03d}" for i in range(1, p + 1)]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "patient_id", _clin_ids(n, "P"))
    df["outcome"] = np.round(y, 3)
    # Add clinical covariates
    df.insert(1, "age", _clin_age(n))
    df.insert(2, "sex", rng.choice(["Male", "Female"], n))
    df.insert(3, "bmi", _clin_bmi(n))
    return df


def make_ml_knn_example() -> pd.DataFrame:
    """Multi-class disease: 4 diseases, 8 biomarkers, partially overlapping clusters."""
    n = 500
    # Keep the example clinically separable, but not so clean that KNN gets perfect ROC curves.
    diseases = ["Rheumatoid Arthritis", "Systemic Lupus", "Sjogren's Syndrome", "Mixed CTD"]
    centers = {
        "Rheumatoid Arthritis": [4.6, 5.2, 4.1, 4.8, 5.4, 4.2, 5.1, 5.6],
        "Systemic Lupus": [5.4, 4.7, 5.8, 4.1, 4.6, 5.7, 4.4, 5.0],
        "Sjogren's Syndrome": [4.1, 5.8, 4.7, 5.3, 4.3, 4.0, 5.6, 4.6],
        "Mixed CTD": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    }
    data = []
    for i in range(n):
        disease = rng.choice(diseases)
        c = centers[disease]
        patient_shift = rng.normal(0, 0.45)
        features = {}
        for j in range(8):
            val = c[j] + patient_shift + rng.normal(0, 1.85)
            features[f"biomarker_{j + 1}"] = np.round(val, 2)
        features["patient_id"] = f"P{str(i + 1).zfill(5)}"
        features["disease"] = disease
        features["age"] = _clin_age(1)[0]
        features["sex"] = rng.choice(["Male", "Female"])
        features["esr"] = np.round(rng.lognormal(3.0, 0.5, 1)[0], 1)
        features["crp"] = np.round(rng.lognormal(1.0, 0.7, 1)[0], 2)
        data.append(features)
    return pd.DataFrame(data)


def make_ml_xgboost_example() -> pd.DataFrame:
    """Rich clinical data: 25 features, non-linear effects, missing patterns, interactions."""
    n = 800
    age = _clin_age(n, 60, 14)
    sex = rng.choice([0, 1], n)
    bmi = _clin_bmi(n)
    sbp = _clin_sbp(n)
    dbp = np.round(sbp * 0.48 + rng.normal(30, 7, n), 0).astype(int)
    glucose = _clin_glucose(n)
    hba1c = np.round(glucose * 0.65 + rng.normal(2.7, 0.4, n), 1)
    cholesterol = _clin_chol(n)
    hdl = np.round(rng.normal(1.22, 0.35, n).clip(0.4, 2.9), 2)
    ldl = np.round(cholesterol - hdl * 0.18 - rng.normal(0.25, 0.3, n), 2)
    triglycerides = np.round(rng.lognormal(0.35, 0.5, n).clip(0.3, 10), 2)
    creatinine = np.round(rng.normal(80, 26, n).clip(35, 260), 1)
    egfr = np.round(170 - 0.85 * age - 0.22 * creatinine + rng.normal(0, 9, n), 1)
    albumin = np.round(rng.normal(40, 5, n).clip(20, 53), 1)
    hemoglobin = np.round(rng.normal(13.5, 1.7, n).clip(7.5, 18.5), 1)
    wbc = np.round(rng.lognormal(2.0, 0.25, n).clip(2.5, 18), 1)
    platelets = np.round(rng.normal(240, 75, n).clip(60, 550), 0).astype(int)
    nlr = np.round(wbc * 1000 / np.clip(platelets * rng.uniform(0.008, 0.015, n), 1, None), 2)
    alt = np.round(rng.lognormal(3.2, 0.35, n).clip(5, 110), 0).astype(int)
    uric_acid = np.round(rng.normal(330, 80, n).clip(100, 600), 1)
    smoking = rng.choice([0, 1], n, p=[0.52, 0.48])
    alcohol = rng.choice([0, 1], n, p=[0.58, 0.42])
    family_hx = rng.choice([0, 1], n, p=[0.62, 0.38])
    exercise = rng.choice([0, 1, 2], n, p=[0.32, 0.42, 0.26])
    sleep_hours = np.round(rng.normal(6.8, 1.3, n).clip(3, 11), 1)
    stress = rng.choice(["Low", "Medium", "High"], n, p=[0.30, 0.45, 0.25])

    # Complex outcome with non-linearities and interactions
    lp = (-4.2
          + 0.038 * age
          + 0.42 * sex
          + 0.055 * bmi
          + 0.016 * sbp
          + 0.26 * hba1c
          - 0.22 * np.log(hdl + 0.01)
          + 0.13 * np.log(triglycerides + 0.01)
          - 0.007 * egfr
          + 0.12 * (nlr > 5)
          + 0.62 * smoking
          + 0.25 * alcohol
          + 0.80 * family_hx
          - 0.18 * (exercise > 0)
          - 0.20 * (exercise == 2)
          + 0.10 * (sleep_hours < 5)
          + 0.25 * (stress == "High")
          + 0.15 * (stress == "Medium")
          + 0.12 * (age > 65) * smoking
          + 0.08 * (bmi > 30) * (hba1c > 7)
          + 0.10 * (sbp > 150) * (age > 60)
          - 0.15 * (hdl > 1.5) * (exercise > 0)
          + rng.normal(0, 0.4, n))
    prob = 1 / (1 + np.exp(-lp))
    outcome = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age, "sex": sex, "bmi": bmi,
        "sbp": sbp, "dbp": dbp,
        "glucose": glucose, "hba1c": hba1c,
        "cholesterol": cholesterol, "hdl": hdl, "ldl": ldl,
        "triglycerides": triglycerides,
        "creatinine": creatinine, "egfr": egfr, "albumin": albumin,
        "hemoglobin": hemoglobin, "wbc": wbc, "platelets": platelets,
        "nlr": nlr, "alt": alt, "uric_acid": uric_acid,
        "smoking": smoking, "alcohol": alcohol,
        "family_history": family_hx,
        "exercise_level": exercise,
        "sleep_hours": sleep_hours, "stress_level": stress,
        "outcome": outcome,
    })


def make_ml_rf_example() -> pd.DataFrame:
    """Random Forest: Survival + classification, complex non-linear structure."""
    n = 700
    age = _clin_age(n, 62, 13)
    sex = rng.choice([0, 1], n)
    bmi = _clin_bmi(n)
    sbp = _clin_sbp(n)
    glucose = _clin_glucose(n)
    hba1c = np.round(glucose * 0.65 + rng.normal(2.7, 0.35, n), 1)
    cholesterol = _clin_chol(n)
    hdl = np.round(rng.normal(1.22, 0.35, n).clip(0.4, 2.9), 2)
    triglycerides = np.round(rng.lognormal(0.35, 0.5, n).clip(0.3, 10), 2)
    creatinine = np.round(rng.normal(80, 26, n).clip(35, 260), 1)
    egfr = np.round(170 - 0.85 * age - 0.22 * creatinine + rng.normal(0, 9, n), 1)
    albumin = np.round(rng.normal(40, 5, n).clip(20, 53), 1)
    hemoglobin = np.round(rng.normal(13.5, 1.7, n).clip(7.5, 18.5), 1)
    nlr = np.round(rng.lognormal(0.8, 0.5, n).clip(0.5, 12), 2)
    smoking = rng.choice([0, 1], n, p=[0.52, 0.48])
    family_hx = rng.choice([0, 1], n, p=[0.62, 0.38])
    stage = rng.choice(["I", "II", "III", "IV"], n, p=[0.20, 0.28, 0.32, 0.20])
    treatment = rng.choice(["Surgery", "Surgery+Chemo", "Chemo+Radio", "Palliative"], n,
                           p=[0.22, 0.30, 0.28, 0.20])
    tumor_size = np.round(rng.lognormal(3.0, 0.55, n).clip(0.3, 14), 1)

    # Non-linear risk model
    risk = (0.032 * age + 0.6 * (stage == "III") + 1.4 * (stage == "IV")
            + 0.12 * tumor_size + 0.18 * (nlr > 4)
            + 0.25 * smoking + 0.22 * (family_hx)
            - 0.008 * albumin - 0.005 * egfr
            + 0.4 * (treatment == "Palliative")
            - 0.35 * (treatment == "Surgery+Chemo")
            + 0.08 * (bmi > 32) * (smoking)
            + 0.12 * (age > 70) * np.isin(stage, ["III", "IV"]).astype(int)
            + rng.normal(0, 0.35, n))

    base_hazard = np.exp(risk - 2.2)
    time = np.round(rng.exponential(1 / np.clip(base_hazard, 0.003, 0.5,), n)).clip(1, 80).astype(int)
    event = rng.choice([0, 1], n, p=[0.38, 0.62])
    # Binary classification with same features
    lp_class = (risk - 0.5 + rng.normal(0, 0.3, n))
    prob_class = 1 / (1 + np.exp(-lp_class))
    class_outcome = (rng.random(n) < prob_class).astype(int)

    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age, "sex": sex, "bmi": bmi,
        "sbp": sbp, "glucose": glucose, "hba1c": hba1c,
        "cholesterol": cholesterol, "hdl": hdl, "triglycerides": triglycerides,
        "creatinine": creatinine, "egfr": egfr, "albumin": albumin,
        "hemoglobin": hemoglobin, "nlr": nlr,
        "smoking": smoking, "family_history": family_hx,
        "stage": stage, "treatment": treatment,
        "treatment_group": np.where(treatment == "Palliative", "Palliative", "Active"),
        "tumor_size_cm": tumor_size,
        "risk_score": np.round(risk, 3),
        "survival_time": time, "survival_event": event,
        "complication": class_outcome,
    })


def make_ml_svm_example() -> pd.DataFrame:
    """Multi-class SVM: 6 biomarkers, non-linear decision boundaries, 3 classes."""
    n = 400
    # Generate spirals-like data for 3 classes
    data = []
    for i in range(n):
        t = rng.uniform(0, 4 * np.pi)
        r = t / (4 * np.pi) * 4 + rng.normal(0, 0.3)
        class_label = i % 3
        angle_offset = class_label * 2 * np.pi / 3
        x1 = r * np.cos(t + angle_offset) + rng.normal(0, 0.2)
        x2 = r * np.sin(t + angle_offset) + rng.normal(0, 0.2)
        data.append({
            "patient_id": f"P{str(i + 1).zfill(5)}",
            "biomarker_1": np.round(x1, 3),
            "biomarker_2": np.round(x2, 3),
            "biomarker_3": np.round(x1 * 0.6 + x2 * 0.4 + rng.normal(0, 0.35), 3),
            "biomarker_4": np.round(x1 * 0.3 - x2 * 0.5 + rng.normal(0, 0.3), 3),
            "biomarker_5": np.round(np.sin(x1) + np.cos(x2 * 0.8) + rng.normal(0, 0.25), 3),
            "biomarker_6": np.round(x1 * x2 * 0.08 + rng.normal(0, 0.4), 3),
            "age": _clin_age(1)[0] if i < n else 0,
            "disease_subtype": ["Subtype A", "Subtype B", "Subtype C"][class_label],
        })
    return pd.DataFrame(data)


def make_ml_dt_example() -> pd.DataFrame:
    """Clinical decision rules: interpretable features, rule-based ground truth."""
    n = 500
    age = _clin_age(n, 56, 15)
    sbp = _clin_sbp(n)
    glucose = _clin_glucose(n)
    hba1c = np.round(glucose * 0.65 + rng.normal(2.7, 0.35, n), 1)
    bmi = _clin_bmi(n)
    cholesterol = _clin_chol(n)
    hdl = np.round(rng.normal(1.22, 0.35, n).clip(0.4, 2.9), 2)
    smoking = rng.choice(["No", "Yes"], n)
    alcohol = rng.choice(["No", "Yes"], n, p=[0.55, 0.45])
    family_hx = rng.choice(["No", "Yes"], n, p=[0.60, 0.40])
    exercise = rng.choice(["None", "Light", "Regular"], n, p=[0.34, 0.38, 0.28])

    # Rule-based risk (perfect for decision tree recovery)
    risk_score = 0
    risk_score += (age > 55).astype(int)
    risk_score += (sbp > 140).astype(int)
    risk_score += (glucose > 7.0).astype(int)
    risk_score += (bmi > 30).astype(int)
    risk_score += (hba1c > 6.5).astype(int)
    risk_score += (cholesterol > 6.0).astype(int)
    risk_score += (hdl < 1.0).astype(int)
    risk_score += (smoking == "Yes").astype(int)
    risk_score += (family_hx == "Yes").astype(int)
    risk_score += (exercise == "None").astype(int)
    # Interaction rules
    risk_score += ((age > 60) & (smoking == "Yes")).astype(int)
    risk_score += ((bmi > 30) & (glucose > 7.0)).astype(int)
    risk_score += ((sbp > 150) & (age > 60)).astype(int)

    high_risk = (risk_score >= 4).astype(int)
    very_high_risk = (risk_score >= 7).astype(int)
    risk_category = np.where(very_high_risk, "Very High",
                             np.where(high_risk, "High", "Standard"))

    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age, "sbp": sbp, "glucose": glucose, "hba1c": hba1c,
        "bmi": bmi, "cholesterol": cholesterol, "hdl": hdl,
        "smoking": smoking, "alcohol": alcohol,
        "family_history": family_hx, "exercise": exercise,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "high_risk": high_risk,
    })


def make_ml_cnn_example() -> pd.DataFrame:
    """1D-CNN: multi-channel physiological time series with overlapping event risk."""
    n_subjects = 250
    timepoints = 60
    channels = ["heart_rate", "blood_pressure", "oxygen_saturation",
                "respiratory_rate", "temperature", "cardiac_output"]
    records = []
    for s in range(1, n_subjects + 1):
        age = rng.integers(32, 82)
        sex = rng.choice(["Male", "Female"])
        bmi = _clin_bmi(1)[0]
        latent_acuity = rng.normal(0, 1) + 0.018 * (age - 56) + 0.045 * (bmi - 25)
        event_prob = 1 / (1 + np.exp(-(0.15 + 0.85 * latent_acuity)))
        label = int(rng.binomial(1, np.clip(event_prob, 0.08, 0.92)))
        trajectory_severity = 0.38 * label + 0.42 * latent_acuity + rng.normal(0, 0.65)
        phase_shift = rng.uniform(0, 2 * np.pi)

        # Subject-specific baselines
        base_hr = rng.normal(78 + 1.8 * latent_acuity, 12)
        base_bp = rng.normal(128 + 2.2 * latent_acuity, 16)
        base_o2 = rng.normal(96.8 - 0.35 * latent_acuity, 2.1)
        base_rr = rng.normal(16.5 + 0.35 * latent_acuity, 2.8)
        base_temp = rng.normal(36.8, 0.35)
        base_co = rng.normal(5.0 - 0.12 * latent_acuity, 0.9)

        # Event-related trajectories are intentionally noisy and overlapping.
        for t in range(timepoints):
            phase = t / max(timepoints - 1, 1)
            late_phase = max(phase - 0.45, 0)
            circadian = np.sin(phase * 2 * np.pi + phase_shift)
            transient = rng.normal(0, 0.35) * (phase > rng.uniform(0.25, 0.85))
            hr_drift = trajectory_severity * (0.035 * t + 1.2 * late_phase) + 1.1 * circadian + transient
            bp_drift = trajectory_severity * (0.020 * t + 1.5 * late_phase) + 1.8 * np.sin(phase * np.pi + phase_shift / 2)
            o2_drift = -trajectory_severity * (0.012 * t + 0.7 * late_phase) + 0.25 * circadian
            rr_drift = trajectory_severity * (0.016 * t + 0.9 * late_phase) + 0.45 * circadian
            temp_drift = trajectory_severity * (0.0018 * t + 0.12 * late_phase) + 0.05 * circadian
            co_drift = -trajectory_severity * (0.0035 * t + 0.22 * late_phase) + 0.08 * circadian

            records.append({
                "subject_id": f"S{str(s).zfill(5)}",
                "time": t,
                "heart_rate": np.round(np.clip(base_hr + hr_drift + rng.normal(0, 4.5), 40, 140), 1),
                "blood_pressure": np.round(np.clip(base_bp + bp_drift + rng.normal(0, 7.0), 60, 200), 1),
                "oxygen_saturation": np.round(np.clip(base_o2 + o2_drift + rng.normal(0, 1.1), 75, 100), 1),
                "respiratory_rate": np.round(np.clip(base_rr + rr_drift + rng.normal(0, 1.9), 6, 35), 1),
                "temperature": np.round(np.clip(base_temp + temp_drift + rng.normal(0, 0.18), 35, 40), 1),
                "cardiac_output": np.round(np.clip(base_co + co_drift + rng.normal(0, 0.45), 2, 9), 2),
                "label": label,
                "age": age, "sex": sex, "bmi": bmi,
            })
    return pd.DataFrame(records)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Integrated Tools Examples
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def make_feature_engineering_example() -> pd.DataFrame:
    """Rich data with missing values, mixed types, outliers, skewed distributions."""
    n = 500
    df = pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": _clin_age(n, 56, 16),
        "sex": rng.choice(["Male", "Female", None], n, p=[0.47, 0.50, 0.03]),
        "bmi": _clin_bmi(n),
        "sbp": _clin_sbp(n),
        "dbp": np.round(rng.normal(80, 11, n).clip(48, 125), 0).astype(int),
        "glucose": _clin_glucose(n),
        "cholesterol": _clin_chol(n),
        "hdl": np.round(rng.normal(1.22, 0.35, n).clip(0.4, 2.9), 2),
        "creatinine": np.round(rng.normal(80, 26, n).clip(35, 260), 1),
        "egfr": np.round(rng.normal(78, 20, n).clip(15, 135), 1),
        "smoking": rng.choice(["Never", "Former", "Current", None], n,
                              p=[0.40, 0.26, 0.28, 0.06]),
        "alcohol": rng.choice(["None", "Light", "Heavy", None], n,
                              p=[0.44, 0.30, 0.20, 0.06]),
        "education": rng.choice(["Primary", "Secondary", "Tertiary", None], n,
                                p=[0.25, 0.42, 0.25, 0.08]),
        "exercise_freq": rng.choice(["None", "1-2x", "3-4x", "5+x", None], n,
                                    p=[0.28, 0.35, 0.22, 0.10, 0.05]),
        "sleep_hours": np.round(rng.normal(6.8, 1.4, n).clip(3, 11), 1),
        "disease_type": rng.choice(["HTN", "DM", "CVD", "COPD", "CKD", "Healthy", None], n,
                                   p=[0.22, 0.18, 0.12, 0.10, 0.08, 0.25, 0.05]),
        "comorbidity_count": rng.poisson(1.5, n).clip(0, 7),
    })

    # Complex missing data patterns
    missing_plan = {
        "bmi": 0.07, "sbp": 0.04, "glucose": 0.09, "cholesterol": 0.08,
        "hdl": 0.10, "creatinine": 0.05, "egfr": 0.06, "sleep_hours": 0.12,
    }
    for col, rate in missing_plan.items():
        n_missing = max(1, int(len(df) * rate))
        missing_idx = rng.choice(df.index, size=n_missing, replace=False)
        df.loc[missing_idx, col] = np.nan

    # Outliers (extreme but plausible clinical values)
    outlier_idx = rng.choice(df.index, size=12, replace=False)
    df.loc[outlier_idx[:3], "sbp"] = rng.choice([215, 228, 235, 52, 48], 3)
    df.loc[outlier_idx[3:6], "bmi"] = rng.choice([48.5, 52.0, 12.5, 13.8], 3)
    df.loc[outlier_idx[6:9], "glucose"] = rng.choice([18.5, 22.0, 1.8, 2.2], 3)
    df.loc[outlier_idx[9:12], "creatinine"] = rng.choice([380, 420, 510], 3)

    # Add a highly skewed variable
    df["crp"] = np.round(rng.lognormal(0.8, 0.9, n).clip(0.05, 80), 2)

    return df


def make_model_comparison_example() -> pd.DataFrame:
    """Rich data for multi-model comparison with known benchmarks."""
    n = 600
    age = _clin_age(n, 58, 13)
    sex = rng.choice([0, 1], n)
    bmi = _clin_bmi(n)
    sbp = _clin_sbp(n)
    glucose = _clin_glucose(n)
    hba1c = np.round(glucose * 0.65 + rng.normal(2.7, 0.35, n), 1)
    cholesterol = _clin_chol(n)
    hdl = np.round(rng.normal(1.22, 0.35, n).clip(0.4, 2.9), 2)
    triglycerides = np.round(rng.lognormal(0.35, 0.5, n).clip(0.3, 10), 2)
    creatinine = np.round(rng.normal(80, 26, n).clip(35, 260), 1)
    egfr = np.round(170 - 0.85 * age - 0.22 * creatinine + rng.normal(0, 9, n), 1)
    smoking = rng.choice([0, 1], n, p=[0.54, 0.46])
    alcohol = rng.choice([0, 1], n, p=[0.60, 0.40])
    family_hx = rng.choice([0, 1], n, p=[0.63, 0.37])

    # Non-linear and interaction-heavy outcome
    lp = (-4.0
          + 0.038 * age + 0.42 * sex + 0.06 * bmi
          + 0.017 * sbp + 0.28 * hba1c
          - 0.23 * np.log(hdl + 0.01)
          + 0.12 * np.log(triglycerides + 0.01)
          - 0.006 * egfr
          + 0.60 * smoking + 0.28 * alcohol + 0.78 * family_hx
          + 0.12 * (age > 62) * smoking
          + 0.09 * (bmi > 32) * (hba1c > 7)
          + 0.08 * np.sin(age / 10)
          + rng.normal(0, 0.4, n))
    prob = 1 / (1 + np.exp(-lp))
    outcome = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age, "sex": sex, "bmi": bmi,
        "sbp": sbp, "glucose": glucose, "hba1c": hba1c,
        "cholesterol": cholesterol, "hdl": hdl, "triglycerides": triglycerides,
        "creatinine": creatinine, "egfr": egfr,
        "smoking": smoking, "alcohol": alcohol, "family_history": family_hx,
        "outcome": outcome,
    })


def make_dim_reduction_example() -> pd.DataFrame:
    """High-dimensional: 100 gene expressions, 5 known subtypes, batch effects."""
    n = 400
    p = 100
    n_subtypes = 5
    # Generate subtype-specific expression patterns
    subtype_centers = rng.normal(0, 1.5, (n_subtypes, p))
    groups = []
    X = np.zeros((n, p))
    for i in range(n):
        subtype = rng.integers(0, n_subtypes)
        groups.append(f"Subtype {chr(65 + subtype)}")
        X[i] = subtype_centers[subtype] + rng.normal(0, 0.7, p)

    # Add batch effect (2 batches)
    batch_effect = rng.normal(0, 0.4, p)
    for i in range(n // 2, n):
        X[i] += batch_effect

    cols = [f"gene_{i:03d}" for i in range(1, p + 1)]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "sample_id", _clin_ids(n, "S"))
    df.insert(1, "subtype", groups)
    df.insert(2, "batch", ["Batch 1"] * (n // 2) + ["Batch 2"] * (n - n // 2))
    df.insert(3, "age", _clin_age(n))
    df.insert(4, "sex", rng.choice(["Male", "Female"], n))
    return df


def make_cluster_example() -> pd.DataFrame:
    """Clustering: 5 natural clusters with varying density, overlap, noise features."""
    n_per = 100
    # 5 clusters with different shapes
    cluster_configs = [
        ("Compact A", 2, 3, 0.5, 0.5),
        ("Compact B", 8, 2, 0.6, 0.6),
        ("Elongated", 5, 7, 1.5, 0.4),
        ("Sparse", 1, 8, 1.2, 1.2),
        ("Dense", 6, 6, 0.35, 0.35),
    ]
    data = []
    for label, cx, cy, sx, sy in cluster_configs:
        for _ in range(n_per):
            # Correlated noise
            rho = rng.uniform(-0.6, 0.6)
            x1 = cx + rng.normal(0, sx)
            x2 = cy + rng.normal(0, sy)
            x3 = cx * 0.5 + cy * 0.3 + rng.normal(0, 0.7)
            x4 = x1 * 0.4 + x2 * 0.6 + rng.normal(0, 0.5)
            x5 = cx * 0.2 - cy * 0.4 + rng.normal(0, 0.8)
            x6 = rng.normal(0, 1.5)  # Noise feature
            data.append({
                "sample_id": f"S{len(data) + 1:05d}",
                "x1": np.round(x1, 2), "x2": np.round(x2, 2),
                "x3": np.round(x3, 2), "x4": np.round(x4, 2),
                "x5": np.round(x5, 2), "x6": np.round(x6, 2),
                "true_cluster": label,
            })
    return pd.DataFrame(data)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Registry
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def make_ml_ridge_example() -> pd.DataFrame:
    """Ridge regression example with correlated clinical predictors."""
    df = make_ml_lasso_example().copy()
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric) >= 3:
        df["renal_metabolic_index"] = np.round(
            0.45 * df[numeric[0]].astype(float)
            + 0.35 * df[numeric[1]].astype(float)
            + 0.20 * df[numeric[2]].astype(float)
            + rng.normal(0, 1.5, len(df)),
            3,
        )
    return df


def make_ldsc_example() -> pd.DataFrame:
    """Compact LDSC-style trait heritability and genetic correlation matrix."""
    traits = ["diabetes", "hypertension", "cad", "stroke", "ckd", "copd"]
    h2 = {
        "diabetes": 0.18,
        "hypertension": 0.24,
        "cad": 0.21,
        "stroke": 0.16,
        "ckd": 0.14,
        "copd": 0.11,
    }
    corr = {
        "diabetes": [1.00, 0.42, 0.36, 0.30, 0.33, 0.12],
        "hypertension": [0.42, 1.00, 0.48, 0.41, 0.28, 0.10],
        "cad": [0.36, 0.48, 1.00, 0.46, 0.24, 0.16],
        "stroke": [0.30, 0.41, 0.46, 1.00, 0.21, 0.11],
        "ckd": [0.33, 0.28, 0.24, 0.21, 1.00, 0.09],
        "copd": [0.12, 0.10, 0.16, 0.11, 0.09, 1.00],
    }
    rows = []
    for trait in traits:
        row = {
            "trait": trait,
            "h2": round(h2[trait] + rng.normal(0, 0.008), 3),
            "h2_se": round(float(np.clip(rng.normal(0.035, 0.006), 0.015, 0.070)), 3),
        }
        row.update({name: corr[trait][i] for i, name in enumerate(traits)})
        rows.append(row)
    return pd.DataFrame(rows)


def make_target_trial_emulation_example() -> pd.DataFrame:
    """Observational cohort shaped for target trial emulation with IPTW."""
    n = 720
    age = _clin_age(n, 64, 11)
    sex = rng.choice(["Male", "Female"], n)
    bmi = _clin_bmi(n, 27.0, 5.0)
    sbp = _clin_sbp(n, 138, 18)
    hba1c = np.round(rng.normal(7.1, 1.2, n).clip(4.8, 12.8), 1)
    egfr = np.round(rng.normal(76, 20, n).clip(18, 130), 1)
    baseline_risk = np.round(
        0.025 * age + 0.040 * bmi + 0.020 * sbp + 0.32 * hba1c - 0.012 * egfr
        + 0.35 * (sex == "Male") + rng.normal(0, 0.6, n),
        3,
    )
    ps_linear = -2.3 + 0.55 * baseline_risk + 0.22 * (sex == "Male") + 0.018 * sbp - 0.010 * egfr
    ps = 1 / (1 + np.exp(-ps_linear))
    treatment = (rng.random(n) < ps).astype(int)
    followup_months = rng.choice([6, 9, 12, 15, 18, 24], n, p=[0.08, 0.10, 0.46, 0.14, 0.12, 0.10])
    event_logit = -3.5 + 0.68 * treatment + 0.62 * baseline_risk + 0.018 * age + 0.18 * (hba1c > 8)
    event_prob = 1 / (1 + np.exp(-event_logit))
    event_12m = (rng.random(n) < event_prob).astype(int)
    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "sbp": sbp,
        "hba1c": hba1c,
        "egfr": egfr,
        "baseline_risk": baseline_risk,
        "treatment": treatment,
        "treatment_group": np.where(treatment == 1, "New treatment", "Usual care"),
        "followup_months": followup_months,
        "event_12m": event_12m,
    })


def make_doubly_robust_aipw_example() -> pd.DataFrame:
    """Binary treatment cohort for AIPW doubly robust causal estimation."""
    n = 680
    age = _clin_age(n, 59, 13)
    sex = rng.choice(["Male", "Female"], n)
    bmi = _clin_bmi(n)
    baseline_score = np.round(rng.normal(48, 12, n).clip(10, 90), 1)
    comorbidity_count = rng.poisson(1.7, n).clip(0, 7)
    crp = np.round(rng.lognormal(1.0, 0.75, n).clip(0.1, 90), 2)
    albumin = np.round(rng.normal(39, 5, n).clip(20, 52), 1)
    ps_linear = -2.0 + 0.025 * age + 0.045 * bmi + 0.018 * baseline_score + 0.28 * comorbidity_count + 0.20 * (sex == "Male")
    ps = 1 / (1 + np.exp(-ps_linear))
    treatment = (rng.random(n) < ps).astype(int)
    potential_gain = 4.2 + 0.05 * baseline_score - 0.35 * comorbidity_count + rng.normal(0, 1.2, n)
    y0 = 32 + 0.32 * baseline_score + 0.08 * age - 0.25 * albumin + 0.18 * crp + rng.normal(0, 5.0, n)
    y1 = y0 + potential_gain
    outcome = np.where(treatment == 1, y1, y0)
    responder = (outcome > np.quantile(outcome, 0.58)).astype(int)
    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "baseline_score": baseline_score,
        "comorbidity_count": comorbidity_count,
        "crp": crp,
        "albumin": albumin,
        "treatment": treatment,
        "outcome": np.round(outcome, 2),
        "responder": responder,
    })


def make_ml_stacking_example() -> pd.DataFrame:
    """Clinical tabular prediction data for stacked ensemble modelling."""
    n = 760
    age = _clin_age(n, 61, 12)
    sex = rng.choice([0, 1], n)
    bmi = _clin_bmi(n)
    sbp = _clin_sbp(n, 136, 18)
    hba1c = np.round(rng.normal(6.8, 1.1, n).clip(4.5, 12), 1)
    egfr = np.round(rng.normal(74, 21, n).clip(15, 135), 1)
    nlr = np.round(rng.lognormal(0.85, 0.55, n).clip(0.4, 18), 2)
    albumin = np.round(rng.normal(39, 5, n).clip(20, 54), 1)
    prior_admission = rng.poisson(0.8, n).clip(0, 6)
    lp = (
        -5.2 + 0.035 * age + 0.34 * sex + 0.045 * bmi + 0.018 * sbp
        + 0.42 * hba1c - 0.010 * egfr + 0.25 * np.log1p(nlr)
        - 0.045 * albumin + 0.38 * prior_admission
        + 0.16 * (age > 70) * (hba1c > 7.5)
        + rng.normal(0, 0.35, n)
    )
    prob = 1 / (1 + np.exp(-lp))
    adverse_event = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "sbp": sbp,
        "hba1c": hba1c,
        "egfr": egfr,
        "nlr": nlr,
        "albumin": albumin,
        "prior_admission": prior_admission,
        "adverse_event": adverse_event,
    })


def make_ml_mlp_example() -> pd.DataFrame:
    """Tabular deep-learning style acute-care prediction dataset."""
    n = 720
    age = _clin_age(n, 58, 15)
    sex = rng.choice([0, 1], n)
    bmi = _clin_bmi(n)
    heart_rate = np.round(rng.normal(86, 18, n).clip(40, 160), 1)
    spo2 = np.round(rng.normal(95, 4, n).clip(70, 100), 1)
    respiratory_rate = np.round(rng.normal(19, 5, n).clip(7, 42), 1)
    creatinine = np.round(rng.normal(86, 31, n).clip(30, 360), 1)
    lactate = np.round(rng.lognormal(0.45, 0.55, n).clip(0.4, 12), 2)
    crp = np.round(rng.lognormal(1.4, 0.8, n).clip(0.2, 160), 2)
    systolic_bp = np.round(rng.normal(122, 22, n).clip(65, 220), 1)
    temp = np.round(rng.normal(37.1, 0.8, n).clip(34.5, 41.5), 1)
    lp = (
        0.2 + 0.020 * age + 0.22 * sex + 0.025 * heart_rate - 0.075 * spo2
        + 0.050 * respiratory_rate + 0.004 * creatinine + 0.55 * lactate
        + 0.008 * crp - 0.018 * systolic_bp + 0.35 * np.maximum(temp - 37.5, 0)
        + 0.12 * (lactate > 2.5) * (spo2 < 93)
        + rng.normal(0, 0.45, n)
    )
    prob = 1 / (1 + np.exp(-lp))
    icu_transfer = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "respiratory_rate": respiratory_rate,
        "creatinine": creatinine,
        "lactate": lactate,
        "crp": crp,
        "systolic_bp": systolic_bp,
        "temperature": temp,
        "icu_transfer": icu_transfer,
    })


def make_ml_elastic_net_example() -> pd.DataFrame:
    """Elastic Net regression example with correlated renal-metabolic predictors."""
    n = 620
    age = _clin_age(n, 61, 12)
    sex = rng.choice([0, 1], n)
    bmi = _clin_bmi(n, 27.5, 5.2)
    sbp = _clin_sbp(n, 136, 18)
    hba1c = np.round(rng.normal(7.0, 1.2, n).clip(4.5, 12.5), 1)
    egfr = np.round(rng.normal(78, 21, n).clip(15, 135), 1)
    albumin = np.round(rng.normal(40, 5, n).clip(20, 54), 1)
    crp = np.round(rng.lognormal(0.9, 0.7, n).clip(0.1, 90), 2)
    uacr = np.round(rng.lognormal(3.1, 0.9, n).clip(2, 1800), 1)
    renal_decline_score = np.round(
        18 + 0.10 * age + 0.18 * bmi + 0.045 * sbp + 1.9 * hba1c
        - 0.18 * egfr - 0.25 * albumin + 0.35 * np.log1p(crp)
        + 0.018 * np.log1p(uacr) * age + 1.1 * sex + rng.normal(0, 5.0, n),
        2,
    )
    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age, "sex": sex, "bmi": bmi, "sbp": sbp,
        "hba1c": hba1c, "egfr": egfr, "albumin": albumin,
        "crp": crp, "uacr": uacr,
        "renal_decline_score": renal_decline_score,
    })


def make_ml_deepsurv_example() -> pd.DataFrame:
    """Oncology survival data for DeepSurv-style neural Cox risk modelling."""
    n = 640
    age = _clin_age(n, 63, 11)
    sex = rng.choice([0, 1], n)
    bmi = _clin_bmi(n, 24.8, 4.2)
    albumin = np.round(rng.normal(38, 6, n).clip(18, 53), 1)
    ldh = np.round(rng.lognormal(5.2, 0.42, n).clip(80, 950), 0)
    nlr = np.round(rng.lognormal(0.9, 0.55, n).clip(0.5, 18), 2)
    tumor_size = np.round(rng.lognormal(2.8, 0.55, n).clip(0.4, 14), 1)
    stage_num = rng.choice([1, 2, 3, 4], n, p=[0.18, 0.26, 0.34, 0.22])
    risk = (
        0.028 * age + 0.18 * sex - 0.035 * bmi - 0.030 * albumin
        + 0.003 * ldh + 0.18 * np.log1p(nlr) + 0.11 * tumor_size
        + 0.42 * (stage_num >= 3) + 0.48 * (stage_num == 4)
        + rng.normal(0, 0.45, n)
    )
    hazard = np.exp(risk - 4.0)
    time_months = np.round(rng.exponential(1 / np.clip(hazard, 0.004, 0.4), n)).clip(1, 96).astype(int)
    censor = rng.integers(12, 96, n)
    event = (time_months <= censor).astype(int)
    observed_time = np.minimum(time_months, censor)
    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age, "sex": sex, "bmi": bmi, "albumin": albumin,
        "ldh": ldh, "nlr": nlr, "tumor_size_cm": tumor_size,
        "stage_num": stage_num,
        "time_months": observed_time,
        "event": event,
    })


def make_ml_deephit_example() -> pd.DataFrame:
    """Competing-risk survival data for DeepHit-style discrete-time risk prediction."""
    n = 680
    age = _clin_age(n, 60, 13)
    sex = rng.choice([0, 1], n)
    bmi = _clin_bmi(n, 26.0, 4.8)
    egfr = np.round(rng.normal(76, 20, n).clip(15, 135), 1)
    hba1c = np.round(rng.normal(6.9, 1.1, n).clip(4.5, 12.5), 1)
    albumin = np.round(rng.normal(39, 5, n).clip(20, 54), 1)
    nlr = np.round(rng.lognormal(0.8, 0.55, n).clip(0.4, 16), 2)
    lp_cvd = -4.2 + 0.032 * age + 0.42 * sex + 0.025 * bmi + 0.25 * hba1c - 0.010 * egfr
    lp_renal = -4.0 + 0.018 * age + 0.35 * hba1c - 0.030 * egfr - 0.035 * albumin + 0.18 * np.log1p(nlr)
    p_cvd = 1 / (1 + np.exp(-lp_cvd))
    p_renal = 1 / (1 + np.exp(-lp_renal))
    draw = rng.random(n)
    event_type = np.where(draw < p_cvd, 1, np.where(draw < p_cvd + p_renal, 2, 0))
    base_time = rng.exponential(28, n) + 3
    time_months = np.round(base_time * np.where(event_type == 0, 1.7, np.where(event_type == 1, 0.9, 1.1))).clip(1, 72).astype(int)
    return pd.DataFrame({
        "patient_id": _clin_ids(n),
        "age": age, "sex": sex, "bmi": bmi, "egfr": egfr,
        "hba1c": hba1c, "albumin": albumin, "nlr": nlr,
        "time_months": time_months,
        "event_type": event_type,
    })


def make_marginal_structural_model_example() -> pd.DataFrame:
    """Longitudinal treatment data with time-varying confounding for MSM."""
    n_subjects = 260
    records = []
    for i in range(1, n_subjects + 1):
        age = rng.integers(35, 84)
        sex = rng.choice(["Male", "Female"])
        baseline_severity = rng.normal(45, 11)
        prior_treatment = 0
        for month in [0, 3, 6, 9, 12]:
            time_varying_severity = baseline_severity + 0.8 * month - 3.5 * prior_treatment + rng.normal(0, 5)
            ps = 1 / (1 + np.exp(-(-2.0 + 0.035 * age + 0.050 * time_varying_severity + 0.45 * prior_treatment)))
            treatment = int(rng.random() < ps)
            outcome = (
                55 + 0.42 * baseline_severity + 0.35 * time_varying_severity
                - 5.2 * treatment - 2.0 * prior_treatment + 0.06 * age
                + rng.normal(0, 6)
            )
            records.append({
                "subject_id": f"S{i:05d}",
                "month": month,
                "age": age,
                "sex": sex,
                "baseline_severity": round(baseline_severity, 2),
                "time_varying_severity": round(time_varying_severity, 2),
                "prior_treatment": prior_treatment,
                "treatment": treatment,
                "outcome": round(outcome, 2),
            })
            prior_treatment = treatment
    return pd.DataFrame(records)


EXAMPLE_MAKERS = {
    "gee_example": make_gee_example,
    "propensity_score_example": make_propensity_score_example,
    "sensitivity_analysis_example": make_sensitivity_analysis_example,
    "counterfactual_example": make_counterfactual_example,
    "survival_advanced_example": make_survival_advanced_example,
    "markov_model_example": make_markov_model_example,
    "bayesian_example": make_bayesian_example,
    "latin_square_example": make_latin_square_example,
    "meta_analysis_example": make_meta_analysis_example,
    "mediation_example": make_mediation_example,
    "mixed_effects_example": make_mixed_effects_example,
    "nhanes_analysis_example": make_nhanes_analysis_example,
    "ldsc_example": make_ldsc_example,
    "target_trial_emulation_example": make_target_trial_emulation_example,
    "doubly_robust_aipw_example": make_doubly_robust_aipw_example,
    "ml_lr_example": make_ml_lr_example,
    "ml_lasso_example": make_ml_lasso_example,
    "ml_ridge_example": make_ml_ridge_example,
    "ml_knn_example": make_ml_knn_example,
    "ml_xgboost_example": make_ml_xgboost_example,
    "ml_rf_example": make_ml_rf_example,
    "ml_svm_example": make_ml_svm_example,
    "ml_dt_example": make_ml_dt_example,
    "ml_cnn_example": make_ml_cnn_example,
    "feature_engineering_example": make_feature_engineering_example,
    "model_comparison_example": make_model_comparison_example,
    "dim_reduction_example": make_dim_reduction_example,
    "cluster_example": make_cluster_example,
    "ml_stacking_example": make_ml_stacking_example,
    "ml_mlp_example": make_ml_mlp_example,
    "ml_elastic_net_example": make_ml_elastic_net_example,
    "ml_deepsurv_example": make_ml_deepsurv_example,
    "ml_deephit_example": make_ml_deephit_example,
    "marginal_structural_model_example": make_marginal_structural_model_example,
}


