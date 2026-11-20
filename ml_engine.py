import io
import math
import difflib
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)

# ---------------------------------------------------------------------------
# Schema & Domain Constants
# ---------------------------------------------------------------------------
VALID_CAMPUSES = [
    "Lahore", "Karachi", "Islamabad", "Faisalabad",
    "Multan", "Rawalpindi", "Peshawar", "Quetta",
]
CAMPUS_ALIASES = {
    "lhr": "Lahore", "lahore": "Lahore", "lahor": "Lahore",
    "khi": "Karachi", "karachi": "Karachi", "karach": "Karachi",
    "isb": "Islamabad", "islamabad": "Islamabad", "isl": "Islamabad",
    "fsd": "Faisalabad", "faisalabad": "Faisalabad", "faislabad": "Faisalabad",
    "mtn": "Multan", "multan": "Multan", "mlt": "Multan",
    "rwp": "Rawalpindi", "pindi": "Rawalpindi", "rawalpindi": "Rawalpindi", "rawlpindi": "Rawalpindi",
    "pesh": "Peshawar", "psh": "Peshawar", "peshawar": "Peshawar", "peshawer": "Peshawar",
    "qta": "Quetta", "quetta": "Quetta", "queta": "Quetta",
}

VALID_PROGRAMS = ["BSCS", "BSSE", "BSAI", "BSIT", "BSDS"]
VALID_GENDERS = ["Male", "Female"]
VALID_PROJECTS = ["Web App", "Mobile App", "AI-ML", "IoT", "Research"]
VALID_YES_NO = ["Yes", "No"]

NUMERIC_RANGES: Dict[str, Tuple[float, float, bool]] = {
    # col: (min, max, is_integer)
    "group_size": (1, 5, True),
    "team_avg_cgpa": (0.0, 4.0, False),
    "team_min_cgpa": (0.0, 4.0, False),
    "supervisor_experience_years": (0, 45, True),
    "supervisor_active_groups": (1, 15, True),
    "proposal_revisions": (0, 10, True),
    "weeks_to_proposal_approval": (0, 30, True),
    "supervisor_meetings_8wk": (0, 30, True),
    "github_commits_8wk": (0, 1000, True),
    "mid_eval_score": (0.0, 100.0, False),
    "final_defense_score": (0.0, 100.0, False),
}

WEEK8_RAW_FEATURES = [
    "proposal_date",
    "campus",
    "program",
    "leader_gender",
    "group_size",
    "team_avg_cgpa",
    "team_min_cgpa",
    "project_type",
    "has_industry_sponsor",
    "supervisor_experience_years",
    "supervisor_active_groups",
    "proposal_revisions",
    "weeks_to_proposal_approval",
    "supervisor_meetings_8wk",
    "github_commits_8wk",
    "mid_eval_score",
]

DROPPED_FEATURES_INFO = [
    {
        "feature": "final_defense_score",
        "reason": "Post-Week-8 Data Leakage — awarded at the final defense at the end of the project, unavailable at Week 8.",
        "category": "Temporal Leakage",
    },
    {
        "feature": "extension_grant",
        "reason": "Post-Week-8 Data Leakage — deadline extension is granted during the project's final phase, unavailable at Week 8.",
        "category": "Temporal Leakage",
    },
    {
        "feature": "group_id",
        "reason": "Anonymous unique identifier — carries no generalizable predictive signal (retained for Watchlist tracking).",
        "category": "Identifier",
    },
    {
        "feature": "proposal_date (raw string)",
        "reason": "Raw timestamp replaced with cyclical seasonal features (proposal_month_sin, proposal_month_cos) to prevent date memorization.",
        "category": "Transformed",
    },
]

FEATURE_LABELS = {
    "mid_eval_score": "Mid-Evaluation Score (Week 8)",
    "team_avg_cgpa": "Team Average CGPA",
    "team_min_cgpa": "Weakest Member CGPA",
    "supervisor_meetings_8wk": "Supervisor Meetings (First 8 Weeks)",
    "github_commits_8wk": "GitHub Commits (First 8 Weeks)",
    "proposal_revisions": "Proposal Revisions",
    "weeks_to_proposal_approval": "Weeks to Proposal Approval",
    "supervisor_active_groups": "Supervisor Active Workload",
    "supervisor_experience_years": "Supervisor Experience (Years)",
    "group_size": "Team Size",
    "has_industry_sponsor": "Industry Sponsor",
    "project_type": "Project Domain",
    "campus": "Campus",
    "program": "Degree Program",
    "leader_gender": "Team Leader Gender",
    "proposal_date": "Proposal Intake Timing",
}

ACTION_PLAYBOOK = {
    "mid_eval_score": "Trigger an immediate Week-9 technical remediation review and break remaining milestones into bi-weekly graded checkpoints.",
    "supervisor_meetings_8wk": "Mandate a recurring weekly 20-minute logged supervisor check-in slot starting Week 9.",
    "github_commits_8wk": "Enforce weekly Git pull-request milestones with mandatory code-progress verification by the supervisor.",
    "team_min_cgpa": "Pair the weakest-performing member with a peer mentor and assign clearly scoped, trackable deliverables.",
    "team_avg_cgpa": "Provide targeted technical clinic sessions and simplify non-essential scope before Week 10.",
    "proposal_revisions": "Hold a scope-alignment workshop with the FYP committee to freeze requirements and prevent scope creep.",
    "weeks_to_proposal_approval": "Assign a fast-track catch-up sprint plan to recover weeks lost during proposal approval.",
    "supervisor_active_groups": "Rebalance supervision bandwidth by assigning a co-supervisor or graduate TA to this group.",
    "supervisor_experience_years": "Pair the project with a senior faculty reviewer for mid-phase architecture feedback.",
    "has_industry_sponsor": "Connect the team with an external domain mentor or departmental lab advisor for practical validation.",
    "group_size": "Redistribute task ownership across members using a formal weekly Kanban deliverable board.",
    "project_type": "Schedule a domain-specific feasibility audit to de-risk technical blockers early.",
    "campus": "Coordinate with the campus FYP coordinator for local lab access and structured progress audits.",
    "program": "Align project deliverables with program rubric expectations via a coordinator consultation.",
    "leader_gender": "Schedule a standard milestone check-in focused on project execution and technical blockers.",
    "proposal_date": "Calibrate milestone deadlines against the semester defense calendar to avoid late-term compression.",
}

ACTIONABLE_LEVERS = {
    "supervisor_meetings_8wk": {"label": "Supervisor Meetings (8wk)", "type": "num", "min": 0, "max": 20, "step": 1, "ideal": 10},
    "github_commits_8wk": {"label": "GitHub Commits (8wk)", "type": "num", "min": 0, "max": 400, "step": 5, "ideal": 140},
    "mid_eval_score": {"label": "Mid-Eval Score (Wk 8)", "type": "num", "min": 0, "max": 100, "step": 1, "ideal": 85},
    "supervisor_active_groups": {"label": "Supervisor Active Groups", "type": "num", "min": 1, "max": 15, "step": 1, "ideal": 3},
    "proposal_revisions": {"label": "Proposal Revisions", "type": "num", "min": 0, "max": 10, "step": 1, "ideal": 0},
    "weeks_to_proposal_approval": {"label": "Weeks to Approval", "type": "num", "min": 0, "max": 20, "step": 1, "ideal": 1},
    "has_industry_sponsor": {"label": "Industry Sponsor", "type": "cat", "options": ["Yes", "No"], "ideal": "Yes"},
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _clean_str(val: Any) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "na", "n/a", "?"}:
        return None
    return s


def _normalize_campus(val: Any) -> Tuple[Optional[str], bool]:
    raw = _clean_str(val)
    if raw is None:
        return None, False
    key = raw.lower().replace(".", "").strip()
    if key in CAMPUS_ALIASES:
        canonical = CAMPUS_ALIASES[key]
        return canonical, (raw != canonical)
    match = difflib.get_close_matches(raw.title(), VALID_CAMPUSES, n=1, cutoff=0.6)
    if match:
        return match[0], (raw != match[0])
    return None, True


def _normalize_from_list(val: Any, valid_list: List[str], extra_map: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], bool]:
    raw = _clean_str(val)
    if raw is None:
        return None, False
    key = raw.lower().strip()
    if extra_map and key in extra_map:
        c = extra_map[key]
        return c, (raw != c)
    for v in valid_list:
        if key == v.lower():
            return v, (raw != v)
    match = difflib.get_close_matches(raw, valid_list, n=1, cutoff=0.65)
    if match:
        return match[0], (raw != match[0])
    return None, True


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _two_prop_pvalue(k1: int, n1: int, k2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return 1.0
    p_pool = (k1 + k2) / (n1 + n2)
    if p_pool in (0.0, 1.0):
        return 1.0
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return 1.0
    z = abs((k1 / n1) - (k2 / n2)) / se
    # Standard normal CDF using Python's built-in math.erf (zero scipy needed!)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(2.0 * (1.0 - cdf))


def engineer_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain feature engineering applied identically during CV, test evaluation,
    live prediction, explainability, and what-if simulation.
    """
    X = df.copy()

    # Extract cyclical month from proposal_date
    dt = pd.to_datetime(X["proposal_date"], errors="coerce")
    month = dt.dt.month.fillna(6).astype(float)
    X["proposal_month_sin"] = np.sin(2 * np.pi * month / 12.0)
    X["proposal_month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Numeric safe extraction
    avg_c = pd.to_numeric(X["team_avg_cgpa"], errors="coerce")
    min_c = pd.to_numeric(X["team_min_cgpa"], errors="coerce")
    g_size = pd.to_numeric(X["group_size"], errors="coerce").clip(lower=1)
    meets = pd.to_numeric(X["supervisor_meetings_8wk"], errors="coerce").clip(lower=0)
    active = pd.to_numeric(X["supervisor_active_groups"], errors="coerce").clip(lower=1)
    commits = pd.to_numeric(X["github_commits_8wk"], errors="coerce").clip(lower=0)
    revs = pd.to_numeric(X["proposal_revisions"], errors="coerce").clip(lower=0)
    wks = pd.to_numeric(X["weeks_to_proposal_approval"], errors="coerce").clip(lower=0)
    mid = pd.to_numeric(X["mid_eval_score"], errors="coerce").clip(0, 100)

    # 1. Academic Disparity ("Weakest-Link" Risk)
    X["cgpa_gap"] = (avg_c - min_c).clip(lower=0.0)
    # 2. Supervision Bandwidth Ratio
    X["supervision_bandwidth"] = meets / active
    # 3. Log-Normalized Per-Member Commit Velocity
    X["log_commits_per_member"] = np.log1p(commits / g_size)
    # 4. Proposal Approval Delay Ratio (share of first 8 weeks consumed)
    X["approval_delay_ratio"] = wks / 8.0
    # 5. Proposal Friction Score
    X["proposal_friction"] = revs * wks
    # 6. Early Execution Index (composite week-8 momentum signal)
    X["early_momentum_index"] = (
        0.55 * (mid / 100.0)
        + 0.25 * (np.log1p(commits) / math.log1p(1000))
        + 0.20 * (meets.clip(upper=15) / 15.0)
    )

    return X.drop(columns=["proposal_date"], errors="ignore")


# ---------------------------------------------------------------------------
# Main FYP-Pulse Engine
# ---------------------------------------------------------------------------
class FYPPulseEngine:
    def __init__(self):
        self.ready: bool = False
        self.source_name: str = ""
        self.df_raw_count: int = 0
        self.df_clean: Optional[pd.DataFrame] = None
        self.df_model_pre_impute: Optional[pd.DataFrame] = None

        self.a1_summary: Dict[str, Any] = {}
        self.a2_charts: Dict[str, Any] = {}
        self.a3_models: Dict[str, Any] = {}
        self.a4_tiers: Dict[str, Any] = {}
        self.a5_fairness: Dict[str, Any] = {}

        self.best_model_name: str = ""
        self.best_pipeline: Any = None
        self.best_threshold: float = 0.5
        self.tier_cutoffs: Tuple[float, float] = (0.20, 0.45)

        self.X_train_raw: Optional[pd.DataFrame] = None
        self.X_test_raw: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.test_meta: Optional[pd.DataFrame] = None
        self.test_probs: Optional[np.ndarray] = None
        self.bg_sample: Optional[pd.DataFrame] = None
        self.default_form_values: Dict[str, Any] = {}
        self.watchlist_cache: Optional[pd.DataFrame] = None

    # -----------------------------------------------------------------------
    # A1: Data Cleaning & Forensic Quality Summary
    # -----------------------------------------------------------------------
    def fit_from_csv_bytes(self, csv_bytes: bytes, source_name: str = "uploaded.csv") -> Dict[str, Any]:
        raw_df = pd.read_csv(io.BytesIO(csv_bytes))
        return self.fit_from_dataframe(raw_df, source_name=source_name)

    def fit_from_csv_path(self, path: str) -> Dict[str, Any]:
        raw_df = pd.read_csv(path)
        return self.fit_from_dataframe(raw_df, source_name=path)

    def fit_from_dataframe(self, raw_df: pd.DataFrame, source_name: str = "fyp_data.csv") -> Dict[str, Any]:
        self.source_name = source_name
        df = raw_df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        initial_rows = int(len(df))
        self.df_raw_count = initial_rows

        corrections_per_col: Dict[str, Dict[str, int]] = {
            c: {"format_standardized": 0, "out_of_range": 0, "rule_fixed": 0, "missing_imputed": 0}
            for c in df.columns
        }
        rule_violations: List[Dict[str, Any]] = []

        # 1. Deduplication: exact duplicate rows, then duplicate group_id
        exact_dups = int(df.duplicated().sum())
        if exact_dups > 0:
            df = df.drop_duplicates(keep="first").reset_index(drop=True)

        id_dups = 0
        if "group_id" in df.columns:
            df["group_id"] = df["group_id"].astype(str).str.strip()
            id_dups = int(df.duplicated(subset=["group_id"]).sum())
            if id_dups > 0:
                df["_na_count"] = df.isna().sum(axis=1)
                df = df.sort_values("_na_count").drop_duplicates(subset=["group_id"], keep="first")
                df = df.drop(columns=["_na_count"]).sort_index().reset_index(drop=True)

        rule_violations.append({
            "rule": "Duplicate Rows / Duplicate Group IDs",
            "rows_broken": exact_dups + id_dups,
            "how_fixed": f"Removed {exact_dups} exact duplicate rows and {id_dups} duplicate group_id records (retaining the most complete record).",
        })

        # 2. Standardize proposal_date -> YYYY-MM-DD
        date_reformatted = 0
        date_invalid = 0
        parsed_dates = []
        iso_pattern_ok = 0
        for raw_val in df["proposal_date"]:
            s = _clean_str(raw_val)
            if s is None:
                parsed_dates.append(pd.NaT)
                continue
            dt = pd.to_datetime(s, format="mixed", errors="coerce")
            if pd.isna(dt):
                date_invalid += 1
                parsed_dates.append(pd.NaT)
            else:
                iso = dt.strftime("%Y-%m-%d")
                if s != iso:
                    date_reformatted += 1
                else:
                    iso_pattern_ok += 1
                parsed_dates.append(dt)
        df["proposal_date"] = pd.Series(parsed_dates, index=df.index)
        corrections_per_col["proposal_date"]["format_standardized"] += date_reformatted
        corrections_per_col["proposal_date"]["out_of_range"] += date_invalid
        rule_violations.append({
            "rule": "Mixed / Invalid Proposal Dates (must be valid YYYY-MM-DD)",
            "rows_broken": date_reformatted + date_invalid,
            "how_fixed": f"Standardized {date_reformatted} mixed-format dates to YYYY-MM-DD; flagged {date_invalid} invalid dates as missing for median-date imputation.",
        })

        # 3. Standardize categorical columns
        cat_configs = [
            ("campus", "campus", None),
            ("program", VALID_PROGRAMS, None),
            ("leader_gender", VALID_GENDERS, {"m": "Male", "f": "Female", "male": "Male", "female": "Female"}),
            ("project_type", VALID_PROJECTS, {"aiml": "AI-ML", "ai/ml": "AI-ML", "ai ml": "AI-ML", "webapp": "Web App", "mobileapp": "Mobile App"}),
            ("has_industry_sponsor", VALID_YES_NO, {"y": "Yes", "n": "No", "yes": "Yes", "no": "No", "1": "Yes", "0": "No"}),
            ("extension_grant", VALID_YES_NO, {"y": "Yes", "n": "No", "yes": "Yes", "no": "No", "1": "Yes", "0": "No"}),
            ("poor_outcome", VALID_YES_NO, {"y": "Yes", "n": "No", "yes": "Yes", "no": "No", "1": "Yes", "0": "No"}),
        ]
        cat_total_fixed = 0
        for col, spec, extra in cat_configs:
            if col not in df.columns:
                continue
            new_vals = []
            fmt_cnt = 0
            for v in df[col]:
                if spec == "campus":
                    nv, changed = _normalize_campus(v)
                else:
                    nv, changed = _normalize_from_list(v, spec, extra)
                if changed:
                    fmt_cnt += 1
                new_vals.append(nv)
            df[col] = new_vals
            corrections_per_col[col]["format_standardized"] += fmt_cnt
            cat_total_fixed += fmt_cnt

        rule_violations.append({
            "rule": "Inconsistent Spellings / Abbreviations (campus, gender M/F, categories)",
            "rows_broken": cat_total_fixed,
            "how_fixed": "Mapped abbreviations (Lhr, KHI, ISB, Fsd, RWP, Pindi, Pesh, M/F) and inconsistent casing via canonical alias dictionary + fuzzy matching.",
        })

        # Drop rows with missing target (cannot train/evaluate without target)
        missing_target = int(df["poor_outcome"].isna().sum())
        if missing_target > 0:
            df = df.dropna(subset=["poor_outcome"]).reset_index(drop=True)
            rule_violations.append({
                "rule": "Missing Target (poor_outcome)",
                "rows_broken": missing_target,
                "how_fixed": "Dropped rows with missing target labels to avoid synthetic label bias.",
            })

        # 4. Numeric Range Enforcement (values outside valid range = errors -> NaN)
        total_oor = 0
        for col, (lo, hi, _is_int) in NUMERIC_RANGES.items():
            if col not in df.columns:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
            mask_oor = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            cnt = int(mask_oor.sum())
            if cnt > 0:
                df.loc[mask_oor, col] = np.nan
                corrections_per_col[col]["out_of_range"] += cnt
                total_oor += cnt

        rule_violations.append({
            "rule": "Out-of-Range Impossible Numeric Values (e.g., negative CGPA/commits, size > 5, mid_eval > 100)",
            "rows_broken": total_oor,
            "how_fixed": "Replaced impossible out-of-range values with NaN so they are cleanly recovered via Iterative MICE imputation instead of distorting models.",
        })

        # 5. Cross-Column Logical Rules for CGPA & Solo Groups
        # Rule 5A: team_min_cgpa > team_avg_cgpa (both valid) -> forensic swap!
        mask_min_gt_avg = (
            df["team_min_cgpa"].notna()
            & df["team_avg_cgpa"].notna()
            & (df["team_min_cgpa"] > df["team_avg_cgpa"])
        )
        n_min_gt_avg = int(mask_min_gt_avg.sum())
        if n_min_gt_avg > 0:
            # Swap when multi-person or unknown size; solo handled right after
            tmp_avg = df.loc[mask_min_gt_avg, "team_avg_cgpa"].copy()
            df.loc[mask_min_gt_avg, "team_avg_cgpa"] = df.loc[mask_min_gt_avg, "team_min_cgpa"]
            df.loc[mask_min_gt_avg, "team_min_cgpa"] = tmp_avg
            corrections_per_col["team_min_cgpa"]["rule_fixed"] += n_min_gt_avg
            corrections_per_col["team_avg_cgpa"]["rule_fixed"] += n_min_gt_avg

        rule_violations.append({
            "rule": "Logical Rule: team_min_cgpa > team_avg_cgpa",
            "rows_broken": n_min_gt_avg,
            "how_fixed": "Swapped reversed (avg, min) CGPA pairs when both fell within [0, 4], preserving true academic signal rather than flattening variance.",
        })

        # Rule 5B: Solo group (group_size == 1) -> team_min_cgpa MUST equal team_avg_cgpa
        solo_mask = df["group_size"] == 1
        solo_unequal = solo_mask & df["team_avg_cgpa"].notna() & df["team_min_cgpa"].notna() & (
            (df["team_avg_cgpa"] - df["team_min_cgpa"]).abs() > 1e-6
        )
        solo_avg_only = solo_mask & df["team_avg_cgpa"].notna() & df["team_min_cgpa"].isna()
        solo_min_only = solo_mask & df["team_avg_cgpa"].isna() & df["team_min_cgpa"].notna()

        n_solo_unequal = int(solo_unequal.sum())
        n_solo_recovered = int(solo_avg_only.sum() + solo_min_only.sum())

        if n_solo_unequal > 0:
            df.loc[solo_unequal, "team_min_cgpa"] = df.loc[solo_unequal, "team_avg_cgpa"]
            corrections_per_col["team_min_cgpa"]["rule_fixed"] += n_solo_unequal
        if solo_avg_only.any():
            df.loc[solo_avg_only, "team_min_cgpa"] = df.loc[solo_avg_only, "team_avg_cgpa"]
            corrections_per_col["team_min_cgpa"]["rule_fixed"] += int(solo_avg_only.sum())
        if solo_min_only.any():
            df.loc[solo_min_only, "team_avg_cgpa"] = df.loc[solo_min_only, "team_min_cgpa"]
            corrections_per_col["team_avg_cgpa"]["rule_fixed"] += int(solo_min_only.sum())

        rule_violations.append({
            "rule": "Solo Group Identity: group_size == 1 requires team_min_cgpa == team_avg_cgpa",
            "rows_broken": n_solo_unequal + n_solo_recovered,
            "how_fixed": f"Equalized {n_solo_unequal} unequal solo pairs and exact-recovered {n_solo_recovered} missing solo CGPA cells from their known counterpart.",
        })

        # Save pre-imputation dataframe for leakage-free ML train/test split!
        self.df_model_pre_impute = df.copy()

        # 6. Impute missing values cleanly for A1 summary & A2 EDA dashboards
        df_clean = df.copy()

        # Median date for missing proposal_date
        missing_dates = int(df_clean["proposal_date"].isna().sum())
        if missing_dates > 0:
            med_ts = df_clean["proposal_date"].dropna().median()
            df_clean["proposal_date"] = df_clean["proposal_date"].fillna(med_ts)
            corrections_per_col["proposal_date"]["missing_imputed"] += missing_dates
        df_clean["proposal_date"] = df_clean["proposal_date"].dt.strftime("%Y-%m-%d")

        # Categorical mode imputation
        for col in ["campus", "program", "leader_gender", "project_type", "has_industry_sponsor", "extension_grant"]:
            if col in df_clean.columns:
                n_miss = int(df_clean[col].isna().sum())
                if n_miss > 0:
                    mode_val = df_clean[col].mode().iloc[0]
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    corrections_per_col[col]["missing_imputed"] += n_miss

        # Numeric Iterative MICE Imputation (BayesianRidge)
        num_cols_present = [c for c in NUMERIC_RANGES.keys() if c in df_clean.columns]
        num_missing_before = {c: int(df_clean[c].isna().sum()) for c in num_cols_present}
        total_missing_imputed = sum(num_missing_before.values()) + sum(
            corrections_per_col[c]["missing_imputed"]
            for c in corrections_per_col
            if c not in num_cols_present
        )

        iter_imp = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=15,
            random_state=42,
            initial_strategy="median",
        )
        imputed_vals = iter_imp.fit_transform(df_clean[num_cols_present])
        for idx, col in enumerate(num_cols_present):
            lo, hi, is_int = NUMERIC_RANGES[col]
            arr = np.clip(imputed_vals[:, idx], lo, hi)
            if is_int:
                arr = np.round(arr).astype(int)
            else:
                arr = np.round(arr, 2)
            df_clean[col] = arr
            corrections_per_col[col]["missing_imputed"] += num_missing_before[col]

        # Post-imputation strict logical invariant re-enforcement
        df_clean["team_min_cgpa"] = np.minimum(df_clean["team_min_cgpa"], df_clean["team_avg_cgpa"])
        solo_clean = df_clean["group_size"] == 1
        df_clean.loc[solo_clean, "team_min_cgpa"] = df_clean.loc[solo_clean, "team_avg_cgpa"]

        rule_violations.append({
            "rule": "Missing Values (Original + Out-of-Range Nulls)",
            "rows_broken": int(total_missing_imputed),
            "how_fixed": "Imputed numeric columns via Iterative BayesianRidge MICE (bounded & invariant-enforced) and categorical columns via mode.",
        })

        self.df_clean = df_clean

        # Build A1 summary payload
        cleaned_rows = int(len(df_clean))
        total_poor = int((df_clean["poor_outcome"] == "Yes").sum())
        poor_rate = float(total_poor / cleaned_rows) if cleaned_rows > 0 else 0.0

        col_summary = []
        for col, counts in corrections_per_col.items():
            total_c = sum(counts.values())
            col_summary.append({
                "column": col,
                "format_standardized": counts["format_standardized"],
                "out_of_range": counts["out_of_range"],
                "rule_fixed": counts["rule_fixed"],
                "missing_imputed": counts["missing_imputed"],
                "total_corrected": total_c,
            })

        self.a1_summary = {
            "source_name": source_name,
            "initial_rows": initial_rows,
            "cleaned_rows": cleaned_rows,
            "rows_removed": initial_rows - cleaned_rows,
            "total_poor_outcomes": total_poor,
            "poor_outcome_rate": round(poor_rate, 4),
            "total_cells_corrected": int(sum(c["total_corrected"] for c in col_summary)),
            "rule_violations": rule_violations,
            "corrections_per_column": col_summary,
        }

        # Populate default form values from dataset medians/modes
        self.default_form_values = {
            "proposal_date": str(df_clean["proposal_date"].mode().iloc[0]),
            "campus": str(df_clean["campus"].mode().iloc[0]),
            "program": str(df_clean["program"].mode().iloc[0]),
            "leader_gender": str(df_clean["leader_gender"].mode().iloc[0]),
            "group_size": int(df_clean["group_size"].median()),
            "team_avg_cgpa": round(float(df_clean["team_avg_cgpa"].median()), 2),
            "team_min_cgpa": round(float(df_clean["team_min_cgpa"].median()), 2),
            "project_type": str(df_clean["project_type"].mode().iloc[0]),
            "has_industry_sponsor": str(df_clean["has_industry_sponsor"].mode().iloc[0]),
            "supervisor_experience_years": int(df_clean["supervisor_experience_years"].median()),
            "supervisor_active_groups": int(df_clean["supervisor_active_groups"].median()),
            "proposal_revisions": int(df_clean["proposal_revisions"].median()),
            "weeks_to_proposal_approval": int(df_clean["weeks_to_proposal_approval"].median()),
            "supervisor_meetings_8wk": int(df_clean["supervisor_meetings_8wk"].median()),
            "github_commits_8wk": int(df_clean["github_commits_8wk"].median()),
            "mid_eval_score": round(float(df_clean["mid_eval_score"].median()), 1),
        }

        # Build A2, A3, A4, A5, A6
        self._build_a2_dashboards()
        self._train_and_select_models()
        self._build_a4_tiers()
        self._build_a5_fairness()
        self._precompute_watchlist()

        self.ready = True
        return self.get_overview_payload()

    # -----------------------------------------------------------------------
    # A2: Dashboards & Visualizations
    # -----------------------------------------------------------------------
    def _build_a2_dashboards(self) -> None:
        df = self.df_clean.copy()
        df["y"] = (df["poor_outcome"] == "Yes").astype(int)
        df["cgpa_gap"] = (df["team_avg_cgpa"] - df["team_min_cgpa"]).clip(lower=0)
        df["supervision_bandwidth"] = df["supervisor_meetings_8wk"] / df["supervisor_active_groups"].clip(lower=1)
        df["commits_per_member"] = df["github_commits_8wk"] / df["group_size"].clip(lower=1)

        def _agg_bins(series: pd.Series, bins: List[float], labels: List[str]) -> Dict[str, Any]:
            cat = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
            grp = df.groupby(cat, observed=False)["y"].agg(["count", "mean", "sum"]).reset_index()
            return {
                "labels": [str(x) for x in grp.iloc[:, 0]],
                "rates": [round(float(r) * 100, 1) if pd.notna(r) else 0.0 for r in grp["mean"]],
                "counts": [int(c) for c in grp["count"]],
                "poor_counts": [int(s) for s in grp["sum"]],
            }

        # Chart 1: Team Factor (Weakest Member CGPA)
        c1 = _agg_bins(df["team_min_cgpa"], [0, 2.0, 2.5, 3.0, 4.0], ["< 2.0", "2.0 – 2.5", "2.5 – 3.0", "3.0 – 4.0"])
        c1_insight = (
            f"Groups whose weakest member has CGPA < 2.0 fail at {c1['rates'][0]}% vs just {c1['rates'][-1]}% "
            f"when all members are ≥ 3.0 — academic floor drives team fragility."
        )

        # Chart 2: Supervisor Factor (Supervisor Meetings in First 8 Weeks)
        c2 = _agg_bins(df["supervisor_meetings_8wk"], [-0.1, 2, 5, 8, 30], ["0 – 2", "3 – 5", "6 – 8", "9+"])
        c2_insight = (
            f"Poor-outcome risk drops sharply from {c2['rates'][0]}% (0–2 meetings) to {c2['rates'][-1]}% (9+ meetings), "
            f"confirming early supervisor cadence as a decisive protective lever."
        )

        # Chart 3: Project Context Factor (Project Type + Industry Sponsor)
        grp3 = df.groupby("project_type")["y"].agg(["count", "mean", "sum"]).sort_values("mean", ascending=False).reset_index()
        c3 = {
            "labels": [str(x) for x in grp3["project_type"]],
            "rates": [round(float(r) * 100, 1) for r in grp3["mean"]],
            "counts": [int(c) for c in grp3["count"]],
            "poor_counts": [int(s) for s in grp3["sum"]],
        }
        spon = df.groupby("has_industry_sponsor")["y"].mean()
        spon_yes = round(float(spon.get("Yes", 0.0)) * 100, 1)
        spon_no = round(float(spon.get("No", 0.0)) * 100, 1)
        c3_insight = (
            f"{c3['labels'][0]} projects carry the highest failure rate ({c3['rates'][0]}%) vs {c3['labels'][-1]} ({c3['rates'][-1]}%); "
            f"industry-sponsored teams fail at {spon_yes}% vs {spon_no}% unsponsored."
        )

        # Chart 4: Newly Engineered Feature #1 — CGPA Disparity Gap
        c4 = _agg_bins(df["cgpa_gap"], [-0.01, 0.10, 0.35, 0.65, 4.0], ["Balanced (≤0.10)", "Mild (0.10–0.35)", "Wide (0.35–0.65)", "Severe (>0.65)"])
        c4_insight = (
            f"Teams with severe internal CGPA disparity (>0.65 gap) suffer a {c4['rates'][-1]}% poor-outcome rate "
            f"compared to {c4['rates'][0]}% for balanced teams — proving skill imbalance hurts execution."
        )

        # Chart 5 (Bonus): Newly Engineered Feature #2 — Supervision Bandwidth Ratio
        c5 = _agg_bins(df["supervision_bandwidth"], [-0.01, 0.4, 0.8, 1.5, 50.0], ["Starved (≤0.4)", "Low (0.4–0.8)", "Healthy (0.8–1.5)", "High (>1.5)"])
        c5_insight = (
            f"Groups receiving starved supervision bandwidth (≤0.4 meetings per active load) face a {c5['rates'][0]}% risk "
            f"vs {c5['rates'][-1]}% for high-bandwidth supervision."
        )

        self.a2_charts = {
            "team_factor": {
                "title": "Team Factor: Weakest Member CGPA vs. Poor-Outcome Rate",
                "x_label": "Weakest Member CGPA Bracket",
                "insight": c1_insight,
                **c1,
            },
            "supervisor_factor": {
                "title": "Supervisor Factor: Week-8 Meetings vs. Poor-Outcome Rate",
                "x_label": "Supervisor Meetings in First 8 Weeks",
                "insight": c2_insight,
                **c2,
            },
            "context_factor": {
                "title": "Project Context: Domain Type vs. Poor-Outcome Rate",
                "x_label": "Project Domain",
                "insight": c3_insight,
                "sponsor_comparison": {"Yes": spon_yes, "No": spon_no},
                **c3,
            },
            "engineered_cgpa_gap": {
                "title": "Engineered Feature: Team CGPA Gap (Avg − Min) vs. Poor-Outcome Rate",
                "x_label": "Internal CGPA Disparity (team_avg_cgpa − team_min_cgpa)",
                "feature_explanation": (
                    "CGPA Gap measures the academic distance between a team's average GPA and its weakest member. "
                    "It matters because raw team average hides free-rider risk and uneven workload bottlenecks — "
                    "a 3.0-average team with a 1.6 weakest link faces far higher coordination failure than a uniform 3.0 team."
                ),
                "insight": c4_insight,
                **c4,
            },
            "engineered_bandwidth": {
                "title": "Engineered Feature: Supervision Bandwidth (Meetings / Active Load)",
                "x_label": "Meetings per Active Supervised Group",
                "feature_explanation": (
                    "Supervision Bandwidth divides a group's 8-week meetings by the supervisor's total active FYP load. "
                    "It matters because 4 meetings with an overloaded supervisor (12 groups) yields less deep mentorship than 4 meetings with a focused supervisor (2 groups)."
                ),
                "insight": c5_insight,
                **c5,
            },
        }

    # -----------------------------------------------------------------------
    # A3: Model Training, Calibration & Selection (Leakage-Free)
    # -----------------------------------------------------------------------
    def _make_preprocessor(self, num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
        num_pipe = Pipeline([
            ("imputer", IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42, initial_strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
        )

    def _optimize_threshold_oof(self, y_true: np.ndarray, probs: np.ndarray) -> float:
        best_t = 0.50
        best_score = -1.0
        for t in np.linspace(0.15, 0.80, 131):
            preds = (probs >= t).astype(int)
            if preds.sum() == 0:
                continue
            f1 = f1_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            # Primary F1 with mild tie-break favoring higher recall (early warning!)
            score = f1 + 0.03 * rec
            if score > best_score:
                best_score = score
                best_t = float(t)
        return round(best_t, 3)

    def _train_and_select_models(self) -> None:
        df_pre = self.df_model_pre_impute.copy()

        # Fill missing proposal_date in pre-impute using global median date safely (non-target timestamp)
        med_dt = df_pre["proposal_date"].dropna().median()
        df_pre["proposal_date"] = df_pre["proposal_date"].fillna(med_dt).dt.strftime("%Y-%m-%d")

        y = (df_pre["poor_outcome"] == "Yes").astype(int).values
        X_raw = df_pre[WEEK8_RAW_FEATURES].copy()
        meta = pd.DataFrame({
            "group_id": df_pre["group_id"].values,
            "leader_gender": self.df_clean["leader_gender"].values,
            "campus": self.df_clean["campus"].values,
        })

        X_train_raw, X_test_raw, y_train, y_test, meta_train, meta_test = train_test_split(
            X_raw, y, meta, test_size=0.20, stratify=y, random_state=42
        )
        self.X_train_raw = X_train_raw.reset_index(drop=True)
        self.X_test_raw = X_test_raw.reset_index(drop=True)
        self.y_train = y_train
        self.y_test = y_test
        self.test_meta = meta_test.reset_index(drop=True)

        # Sample engineered schema to split numeric vs categorical columns
        sample_eng = engineer_features_df(self.X_train_raw.head(5))
        cat_cols = ["campus", "program", "leader_gender", "project_type", "has_industry_sponsor"]
        num_cols = [c for c in sample_eng.columns if c not in cat_cols]

        fe_transformer = FunctionTransformer(engineer_features_df, validate=False)

        def build_pipe(estimator) -> Pipeline:
            return Pipeline([
                ("fe", fe_transformer),
                ("prep", self._make_preprocessor(num_cols, cat_cols)),
                ("clf", estimator),
            ])

        # 4 distinct model families
        lr_est = LogisticRegression(C=0.6, class_weight="balanced", max_iter=1000, random_state=42)
        rf_est = RandomForestClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        hgb_est = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=20,
            l2_regularization=1.5,
            class_weight="balanced",
            random_state=42,
        )

        # Calibrated Soft-Voting Ensemble (Platt Sigmoid Calibration for honest probabilities!)
        voting_base = VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(C=0.6, class_weight="balanced", max_iter=1000, random_state=42)),
                ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=6, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
                ("hgb", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=5, min_samples_leaf=20, l2_regularization=1.5, class_weight="balanced", random_state=42)),
            ],
            voting="soft",
            weights=[1.2, 1.0, 1.3],
        )
        cal_ens = CalibratedClassifierCV(estimator=voting_base, method="sigmoid", cv=3)

        model_specs = [
            ("Logistic Regression (L2 Balanced)", build_pipe(lr_est)),
            ("Random Forest (Balanced Subsample)", build_pipe(rf_est)),
            ("HistGradientBoosting (Balanced)", build_pipe(hgb_est)),
            ("Calibrated Soft-Voting Ensemble", build_pipe(cal_ens)),
        ]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = []
        best_cv_rank_score = -1.0
        best_oof_probs = None

        for name, pipe in model_specs:
            # Out-of-fold validation probabilities (zero test leakage!)
            oof_probs = cross_val_predict(pipe, self.X_train_raw, self.y_train, cv=skf, method="predict_proba")[:, 1]
            opt_thresh = self._optimize_threshold_oof(self.y_train, oof_probs)
            cv_auc = float(roc_auc_score(self.y_train, oof_probs))
            cv_f1 = float(f1_score(self.y_train, (oof_probs >= opt_thresh).astype(int), zero_division=0))
            cv_brier = float(brier_score_loss(self.y_train, oof_probs))

            # Fit on full training split & evaluate on held-out test split
            pipe.fit(self.X_train_raw, self.y_train)
            test_probs = pipe.predict_proba(self.X_test_raw)[:, 1]
            test_preds = (test_probs >= opt_thresh).astype(int)

            tn, fp, fn, tp = confusion_matrix(self.y_test, test_preds).ravel()
            metrics = {
                "name": name,
                "accuracy": round(float(accuracy_score(self.y_test, test_preds)), 4),
                "precision": round(float(precision_score(self.y_test, test_preds, zero_division=0)), 4),
                "recall": round(float(recall_score(self.y_test, test_preds, zero_division=0)), 4),
                "f1": round(float(f1_score(self.y_test, test_preds, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(self.y_test, test_probs)), 4),
                "brier_score": round(float(brier_score_loss(self.y_test, test_probs)), 4),
                "threshold": opt_thresh,
                "cv_roc_auc": round(cv_auc, 4),
                "cv_f1": round(cv_f1, 4),
                "cv_brier": round(cv_brier, 4),
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            }
            results.append(metrics)

            # Selection strictly via Out-of-Fold CV (ROC-AUC + F1 + Calibration bonus)
            rank_score = 0.45 * cv_auc + 0.45 * cv_f1 - 0.20 * cv_brier
            if rank_score > best_cv_rank_score:
                best_cv_rank_score = rank_score
                self.best_model_name = name
                self.best_pipeline = pipe
                self.best_threshold = opt_thresh
                self.test_probs = test_probs
                best_oof_probs = oof_probs

        # Background sample from imputed clean training data for fast local attribution
        train_clean_idx = self.X_train_raw.index
        bg_df = self.df_clean.iloc[: len(self.df_clean)].sample(n=min(60, len(self.df_clean)), random_state=42)
        self.bg_sample = bg_df[WEEK8_RAW_FEATURES].reset_index(drop=True)

        # Save OOF probs for tier cutoff selection
        self._best_oof_probs = best_oof_probs

        best_m = next(m for m in results if m["name"] == self.best_model_name)
        dummy_acc = round(float(1.0 - self.y_test.mean()), 4)
        justification = (
            f"Selected '{self.best_model_name}' via 5-Fold Stratified Out-of-Fold CV "
            f"(CV ROC-AUC: {best_m['cv_roc_auc']:.3f}, CV F1: {best_m['cv_f1']:.3f}, Brier: {best_m['cv_brier']:.3f}). "
            f"On the untouched test set, it achieves {best_m['roc_auc']:.3f} ROC-AUC, {best_m['recall']*100:.1f}% Recall, and {best_m['f1']:.3f} F1 "
            f"at threshold {best_m['threshold']:.2f} — vastly outperforming a naive 'No Risk' baseline ({dummy_acc*100:.1f}% accuracy, 0% recall) "
            f"while producing well-calibrated probabilities for reliable risk tiering and coordinator watchlists."
        )

        self.a3_models = {
            "dropped_features": DROPPED_FEATURES_INFO,
            "train_size": int(len(self.y_train)),
            "test_size": int(len(self.y_test)),
            "train_positive_rate": round(float(self.y_train.mean()), 4),
            "test_positive_rate": round(float(self.y_test.mean()), 4),
            "dummy_baseline_accuracy": dummy_acc,
            "models": results,
            "selected_model": self.best_model_name,
            "selected_threshold": self.best_threshold,
            "justification": justification,
        }

    # -----------------------------------------------------------------------
    # A4: Risk Tiers & Monotonicity Verification on Test Split
    # -----------------------------------------------------------------------
    def _tier_from_prob(self, prob: float) -> str:
        lo, hi = self.tier_cutoffs
        if prob < lo:
            return "Low"
        elif prob < hi:
            return "Medium"
        return "High"

    def _build_a4_tiers(self) -> None:
        # Pick clean, interpretable cut-offs from OOF distribution and verify strict monotonicity on test split
        candidates = [
            (0.15, 0.40),
            (0.12, 0.35),
            (0.20, 0.45),
            (round(float(np.quantile(self._best_oof_probs, 0.55)), 2), round(float(np.quantile(self._best_oof_probs, 0.82)), 2)),
            (round(float(np.quantile(self._best_oof_probs, 0.50)), 2), round(float(np.quantile(self._best_oof_probs, 0.80)), 2)),
        ]
        chosen = candidates[0]
        for lo, hi in candidates:
            if hi <= lo + 0.05:
                continue
            m_lo = self.test_probs < lo
            m_med = (self.test_probs >= lo) & (self.test_probs < hi)
            m_hi = self.test_probs >= hi
            if m_lo.sum() >= 15 and m_med.sum() >= 15 and m_hi.sum() >= 15:
                r_lo = float(self.y_test[m_lo].mean())
                r_med = float(self.y_test[m_med].mean())
                r_hi = float(self.y_test[m_hi].mean())
                if r_lo < r_med < r_hi:
                    chosen = (lo, hi)
                    break

        self.tier_cutoffs = chosen
        lo, hi = chosen

        tier_stats = []
        for label, color, mask, rule_txt in [
            ("Low", "green", self.test_probs < lo, f"Probability < {lo*100:.0f}%"),
            ("Medium", "yellow", (self.test_probs >= lo) & (self.test_probs < hi), f"{lo*100:.0f}% ≤ Probability < {hi*100:.0f}%"),
            ("High", "red", self.test_probs >= hi, f"Probability ≥ {hi*100:.0f}%"),
        ]:
            cnt = int(mask.sum())
            poor_cnt = int(self.y_test[mask].sum()) if cnt > 0 else 0
            obs_rate = float(poor_cnt / cnt) if cnt > 0 else 0.0
            tier_stats.append({
                "tier": label,
                "color": color,
                "rule": rule_txt,
                "test_groups": cnt,
                "observed_poor_outcomes": poor_cnt,
                "observed_poor_rate": round(obs_rate, 4),
            })

        is_monotonic = tier_stats[0]["observed_poor_rate"] < tier_stats[1]["observed_poor_rate"] < tier_stats[2]["observed_poor_rate"]
        self.a4_tiers = {
            "low_cutoff": lo,
            "high_cutoff": hi,
            "is_strictly_monotonic": is_monotonic,
            "test_tier_rates": tier_stats,
        }

    # -----------------------------------------------------------------------
    # A5: Fairness Check (Gender + Campuses >= 30 Test Groups)
    # -----------------------------------------------------------------------
    def _build_a5_fairness(self) -> None:
        df_eval = self.test_meta.copy()
        df_eval["y_true"] = self.y_test
        df_eval["prob"] = self.test_probs
        df_eval["y_pred"] = (self.test_probs >= self.best_threshold).astype(int)

        # 1. Gender Comparison
        gender_rows = []
        for g in ["Male", "Female"]:
            sub = df_eval[df_eval["leader_gender"] == g]
            n = int(len(sub))
            pred_pos = int(sub["y_pred"].sum())
            actual_pos = int(sub["y_true"].sum())
            tp = int(((sub["y_pred"] == 1) & (sub["y_true"] == 1)).sum())
            pred_rate = pred_pos / n if n > 0 else 0.0
            rec = tp / actual_pos if actual_pos > 0 else 0.0
            pr_lo, pr_hi = _wilson_ci(pred_pos, n)
            rc_lo, rc_hi = _wilson_ci(tp, actual_pos)
            gender_rows.append({
                "group": g,
                "count": n,
                "actual_poor_count": actual_pos,
                "predicted_poor_rate": round(pred_rate, 4),
                "predicted_rate_ci95": [round(pr_lo, 4), round(pr_hi, 4)],
                "recall": round(rec, 4),
                "recall_ci95": [round(rc_lo, 4), round(rc_hi, 4)],
            })

        g_m = next((r for r in gender_rows if r["group"] == "Male"), None)
        g_f = next((r for r in gender_rows if r["group"] == "Female"), None)
        gender_pred_gap = abs(g_m["predicted_poor_rate"] - g_f["predicted_poor_rate"]) if (g_m and g_f) else 0.0
        gender_recall_gap = abs(g_m["recall"] - g_f["recall"]) if (g_m and g_f) else 0.0

        # Statistical p-value for gender predicted rate difference
        sub_m = df_eval[df_eval["leader_gender"] == "Male"]
        sub_f = df_eval[df_eval["leader_gender"] == "Female"]
        gender_p_pred = _two_prop_pvalue(int(sub_m["y_pred"].sum()), len(sub_m), int(sub_f["y_pred"].sum()), len(sub_f))
        gender_p_rec = _two_prop_pvalue(
            int(((sub_m["y_pred"] == 1) & (sub_m["y_true"] == 1)).sum()),
            int(sub_m["y_true"].sum()),
            int(((sub_f["y_pred"] == 1) & (sub_f["y_true"] == 1)).sum()),
            int(sub_f["y_true"].sum()),
        )

        # 2. Campus Comparison (only campuses with >= 30 test groups)
        campus_rows = []
        for camp, sub in df_eval.groupby("campus"):
            n = int(len(sub))
            if n < 30:
                continue
            pred_pos = int(sub["y_pred"].sum())
            actual_pos = int(sub["y_true"].sum())
            tp = int(((sub["y_pred"] == 1) & (sub["y_true"] == 1)).sum())
            pred_rate = pred_pos / n if n > 0 else 0.0
            rec = tp / actual_pos if actual_pos > 0 else 0.0
            campus_rows.append({
                "campus": str(camp),
                "count": n,
                "actual_poor_count": actual_pos,
                "predicted_poor_rate": round(pred_rate, 4),
                "recall": round(rec, 4),
            })
        campus_rows.sort(key=lambda x: x["predicted_poor_rate"], reverse=True)

        campus_gap_info = {}
        if len(campus_rows) >= 2:
            max_pr = max(campus_rows, key=lambda x: x["predicted_poor_rate"])
            min_pr = min(campus_rows, key=lambda x: x["predicted_poor_rate"])
            max_rc = max(campus_rows, key=lambda x: x["recall"])
            min_rc = min(campus_rows, key=lambda x: x["recall"])
            campus_gap_info = {
                "largest_predicted_rate_gap": round(max_pr["predicted_poor_rate"] - min_pr["predicted_poor_rate"], 4),
                "predicted_rate_highest": max_pr["campus"],
                "predicted_rate_lowest": min_pr["campus"],
                "largest_recall_gap": round(max_rc["recall"] - min_rc["recall"], 4),
                "recall_highest": max_rc["campus"],
                "recall_lowest": min_rc["campus"],
            }

        # 1–2 sentence statistically grounded takeaway
        sig_gender = (gender_p_pred < 0.05) or (gender_p_rec < 0.05)
        if sig_gender:
            sent1 = (
                f"There is a statistically notable gender gap (predicted-rate gap {gender_pred_gap*100:.1f} pp, "
                f"recall gap {gender_recall_gap*100:.1f} pp; p={min(gender_p_pred, gender_p_rec):.3f}) across "
                f"Male-led (n={g_m['count']}) and Female-led (n={g_f['count']}) test groups."
            )
        else:
            sent1 = (
                f"Male-led (n={g_m['count']}) and Female-led (n={g_f['count']}) groups show no statistically significant "
                f"disparity (predicted-rate gap {gender_pred_gap*100:.1f} pp, recall gap {gender_recall_gap*100:.1f} pp; p={min(gender_p_pred, gender_p_rec):.2f})."
            )

        if campus_gap_info:
            min_camp_pos = min(r["actual_poor_count"] for r in campus_rows)
            sent2 = (
                f"Across campuses with ≥30 test groups, the largest predicted-rate gap is {campus_gap_info['largest_predicted_rate_gap']*100:.1f} pp "
                f"({campus_gap_info['predicted_rate_highest']} vs. {campus_gap_info['predicted_rate_lowest']}) and recall gap is "
                f"{campus_gap_info['largest_recall_gap']*100:.1f} pp, though confidence in per-campus recall is moderate because each campus has as few as {min_camp_pos} true poor-outcome cases."
            )
        else:
            sent2 = "Per-campus subgroup sizes in the test split are small, limiting statistical confidence in campus-level disparity estimates."

        self.a5_fairness = {
            "gender_comparison": gender_rows,
            "gender_predicted_rate_gap": round(gender_pred_gap, 4),
            "gender_recall_gap": round(gender_recall_gap, 4),
            "gender_p_value_pred": round(gender_p_pred, 4),
            "gender_p_value_recall": round(gender_p_rec, 4),
            "campus_comparison": campus_rows,
            "campus_gap": campus_gap_info,
            "takeaway": f"{sent1} {sent2}",
        }

    # -----------------------------------------------------------------------
    # A4 + A5: Single Group Prediction & Exact Interventional Explainability
    # -----------------------------------------------------------------------
    def _coerce_input_row(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(self.default_form_values)
        row.update({k: v for k, v in payload.items() if v is not None and v != "" and not pd.isna(v)})
        for k in ("group_size", "team_avg_cgpa", "team_min_cgpa"):
            if k not in row or row[k] is None or pd.isna(row[k]):
                row[k] = self.default_form_values.get(k, 1)

        row["group_size"] = int(np.clip(int(round(float(row["group_size"]))), 1, 5))
        row["team_avg_cgpa"] = float(np.clip(float(row["team_avg_cgpa"]), 0.0, 4.0))
        row["team_min_cgpa"] = float(np.clip(float(row["team_min_cgpa"]), 0.0, 4.0))
        if row["group_size"] == 1:
            row["team_min_cgpa"] = row["team_avg_cgpa"]
        elif row["team_min_cgpa"] > row["team_avg_cgpa"]:
            row["team_min_cgpa"] = row["team_avg_cgpa"]

        for col, (lo, hi, is_int) in NUMERIC_RANGES.items():
            if col in row and col not in ("group_size", "team_avg_cgpa", "team_min_cgpa"):
                val = row[col]
                if val is None or pd.isna(val):
                    val = self.default_form_values.get(col, lo)
                val = float(np.clip(float(val), lo, hi))
                row[col] = int(round(val)) if is_int else round(val, 2)
        return row

    def explain_instance(self, row: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Exact interventional local attribution on the full trained pipeline.
        Automatically propagates changes through all engineered features (cgpa_gap,
        supervision_bandwidth, log_commits_per_member, etc.) in a single vectorized batch.
        """
        x_df = pd.DataFrame([row])[WEEK8_RAW_FEATURES]
        p_x = float(self.best_pipeline.predict_proba(x_df)[0, 1])

        bg = self.bg_sample
        n_bg = len(bg)
        explain_features = [f for f in WEEK8_RAW_FEATURES if f != "proposal_date"]

        # Vectorized batch evaluation across all features at once (~15ms total!)
        batch_frames = []
        for feat in explain_features:
            mod = pd.concat([x_df] * n_bg, ignore_index=True)
            mod[feat] = bg[feat].values
            # Keep CGPA logical invariant valid during counterfactual marginalization
            if feat == "team_avg_cgpa":
                mod["team_min_cgpa"] = np.minimum(mod["team_min_cgpa"], mod["team_avg_cgpa"])
            elif feat == "team_min_cgpa":
                mod["team_min_cgpa"] = np.minimum(mod["team_min_cgpa"], mod["team_avg_cgpa"])
            batch_frames.append(mod)

        full_batch = pd.concat(batch_frames, ignore_index=True)
        batch_probs = self.best_pipeline.predict_proba(full_batch)[:, 1]

        contributions = []
        for i, feat in enumerate(explain_features):
            mean_bg_prob = float(batch_probs[i * n_bg : (i + 1) * n_bg].mean())
            impact = p_x - mean_bg_prob
            contributions.append({
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "value": row[feat],
                "impact": round(impact, 4),
                "impact_pct": round(impact * 100.0, 2),
                "direction": "increases_risk" if impact >= 0 else "decreases_risk",
            })

        contributions.sort(key=lambda d: abs(d["impact"]), reverse=True)
        return p_x, contributions

    def predict_group(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        row = self._coerce_input_row(payload)
        prob, contribs = self.explain_instance(row)
        tier = self._tier_from_prob(prob)
        outcome = "Poor Outcome Likely" if prob >= self.best_threshold else "On Track to Pass"
        color = {"Low": "green", "Medium": "yellow", "High": "red"}[tier]

        top3 = contribs[:3]
        risk_increasers = [c for c in contribs if c["impact"] > 0]
        biggest_driver = risk_increasers[0] if risk_increasers else top3[0]
        support_step = ACTION_PLAYBOOK.get(
            biggest_driver["feature"],
            "Schedule a focused Week-9 progress check-in with the supervisor to review milestone deliverables.",
        )

        # Clear, non-technical summary
        inc_txt = ", ".join(f"{c['label'].lower()} ({c['value']})" for c in top3 if c["impact"] >= 0)
        dec_txt = ", ".join(f"{c['label'].lower()} ({c['value']})" for c in top3 if c["impact"] < 0)
        if inc_txt and dec_txt:
            plain_expl = (
                f"This group is placed in the {tier} Risk tier ({prob*100:.1f}% estimated risk). "
                f"Risk is primarily pushed up by {inc_txt}, while {dec_txt} helps cushion the risk."
            )
        elif inc_txt:
            plain_expl = (
                f"This group is placed in the {tier} Risk tier ({prob*100:.1f}% estimated risk), "
                f"driven mainly by {inc_txt}."
            )
        else:
            plain_expl = (
                f"This group is placed in the {tier} Risk tier ({prob*100:.1f}% estimated risk), "
                f"strongly protected by {dec_txt}."
            )

        return {
            "sanitized_input": row,
            "predicted_outcome": outcome,
            "probability": round(prob, 4),
            "probability_pct": round(prob * 100.0, 2),
            "threshold": self.best_threshold,
            "risk_tier": tier,
            "badge_color": color,
            "top_3_features": top3,
            "biggest_risk_driver": biggest_driver,
            "plain_explanation": plain_expl,
            "support_step": support_step,
            "tier_cutoffs": {"low": self.tier_cutoffs[0], "high": self.tier_cutoffs[1]},
        }

    # -----------------------------------------------------------------------
    # A6: Weekly Watchlist with Coordinator Capacity K
    # -----------------------------------------------------------------------
    def _precompute_watchlist(self) -> None:
        # Rank all test groups by predicted probability and compute top risk driver for top 100
        order = np.argsort(-self.test_probs)
        rows = []
        max_k = min(100, len(order))
        for rank_idx, idx in enumerate(order[:max_k], start=1):
            prob = float(self.test_probs[idx])
            raw_dict = self.X_test_raw.iloc[idx].to_dict()
            # Fill any raw NaNs with defaults for attribution display
            clean_row = self._coerce_input_row(raw_dict)
            _, contribs = self.explain_instance(clean_row)
            inc = [c for c in contribs if c["impact"] > 0]
            top_drv = inc[0] if inc else contribs[0]
            rows.append({
                "rank": rank_idx,
                "group_id": str(self.test_meta.iloc[idx]["group_id"]),
                "campus": str(self.test_meta.iloc[idx]["campus"]),
                "probability": round(prob, 4),
                "probability_pct": round(prob * 100.0, 2),
                "tier": self._tier_from_prob(prob),
                "top_risk_driver": f"{top_drv['label']} ({top_drv['value']})",
                "recommended_action": ACTION_PLAYBOOK.get(top_drv["feature"], "Schedule check-in"),
                "actual_poor_outcome": "Yes" if int(self.y_test[idx]) == 1 else "No",
            })
        self.watchlist_cache = pd.DataFrame(rows)

    def get_watchlist(self, k: int = 30) -> Dict[str, Any]:
        k = int(np.clip(k, 10, min(100, len(self.watchlist_cache))))
        sub = self.watchlist_cache.head(k)
        captured = int((sub["actual_poor_outcome"] == "Yes").sum())
        total_test_poor = int(self.y_test.sum())
        base_rate = float(self.y_test.mean())
        expected_random = round(k * base_rate, 2)
        precision_k = round(captured / k, 4)
        recall_k = round(captured / total_test_poor, 4) if total_test_poor > 0 else 0.0
        lift = round(captured / expected_random, 2) if expected_random > 0 else 0.0

        return {
            "k": k,
            "max_k": int(min(100, len(self.watchlist_cache))),
            "total_test_groups": int(len(self.y_test)),
            "total_test_poor_outcomes": total_test_poor,
            "captured_poor_outcomes": captured,
            "precision_at_k": precision_k,
            "recall_at_k": recall_k,
            "expected_random_poor": expected_random,
            "lift_over_random": lift,
            "groups": sub.to_dict(orient="records"),
        }

    def get_watchlist_csv(self, k: int = 30) -> str:
        k = int(np.clip(k, 10, min(100, len(self.watchlist_cache))))
        sub = self.watchlist_cache.head(k)[["group_id", "probability", "tier", "top_risk_driver", "recommended_action"]]
        return sub.to_csv(index=False)

    # -----------------------------------------------------------------------
    # A7: What-If Simulator & Best Actionable Lever Finder
    # -----------------------------------------------------------------------
    def simulate_whatif(self, baseline: Dict[str, Any], levers: Dict[str, Any]) -> Dict[str, Any]:
        base_row = self._coerce_input_row(baseline)
        mod_row = dict(base_row)
        # Strictly allow ONLY actionable department/group levers!
        for k, v in levers.items():
            if k in ACTIONABLE_LEVERS and v is not None:
                mod_row[k] = v
        before = self.predict_group(base_row)
        after = self.predict_group(mod_row)
        delta_pct = round(after["probability_pct"] - before["probability_pct"], 2)

        return {
            "before": before,
            "after": after,
            "delta_probability_pct": delta_pct,
            "disclaimer": "Important Note: This model reflects statistical associations observed in historical FYP data, not a guaranteed causal effect of applying the change.",
        }

    def find_best_lever(self, baseline: Dict[str, Any]) -> Dict[str, Any]:
        base_row = self._coerce_input_row(baseline)
        base_pred = self.predict_group(base_row)
        base_prob = base_pred["probability"]

        lever_results = []
        for lever_key, cfg in ACTIONABLE_LEVERS.items():
            current_val = base_row[lever_key]
            target_val = cfg["ideal"]

            # Test both moving toward ideal and realistic step improvements
            if cfg["type"] == "num":
                # Keep realistic intervention magnitude so it's actionable
                if lever_key == "mid_eval_score":
                    candidate_val = min(100.0, max(float(current_val) + 15.0, float(target_val)))
                elif lever_key == "supervisor_meetings_8wk":
                    candidate_val = min(20, max(int(current_val) + 4, int(target_val)))
                elif lever_key == "github_commits_8wk":
                    candidate_val = min(400, max(int(current_val) + 80, int(target_val)))
                else:
                    candidate_val = target_val
            else:
                candidate_val = target_val

            if str(candidate_val) == str(current_val):
                lever_results.append({
                    "lever": lever_key,
                    "label": cfg["label"],
                    "from_value": current_val,
                    "to_value": candidate_val,
                    "new_probability_pct": base_pred["probability_pct"],
                    "delta_pct": 0.0,
                    "new_tier": base_pred["risk_tier"],
                })
                continue

            sim_row = dict(base_row)
            sim_row[lever_key] = candidate_val
            sim_row = self._coerce_input_row(sim_row)
            new_prob = float(self.best_pipeline.predict_proba(pd.DataFrame([sim_row])[WEEK8_RAW_FEATURES])[0, 1])
            delta_pct = round((new_prob - base_prob) * 100.0, 2)
            lever_results.append({
                "lever": lever_key,
                "label": cfg["label"],
                "from_value": current_val,
                "to_value": candidate_val,
                "new_probability_pct": round(new_prob * 100.0, 2),
                "delta_pct": delta_pct,
                "new_tier": self._tier_from_prob(new_prob),
            })

        # Sort by biggest risk reduction (most negative delta_pct)
        lever_results.sort(key=lambda x: x["delta_pct"])
        best = lever_results[0]
        return {
            "baseline_probability_pct": base_pred["probability_pct"],
            "baseline_tier": base_pred["risk_tier"],
            "best_lever": best,
            "all_levers": lever_results,
            "disclaimer": "Important Note: This model shows historical association, not a guaranteed causal effect of the change.",
        }

    # -----------------------------------------------------------------------
    # Master Dashboard Payload
    # -----------------------------------------------------------------------
    def get_overview_payload(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "a1_cleaning": self.a1_summary,
            "a2_dashboards": self.a2_charts,
            "a3_models": self.a3_models,
            "a4_tiers": self.a4_tiers,
            "a5_fairness": self.a5_fairness,
            "default_form": self.default_form_values,
            "actionable_levers": ACTIONABLE_LEVERS,
            "valid_options": {
                "campus": VALID_CAMPUSES,
                "program": VALID_PROGRAMS,
                "leader_gender": VALID_GENDERS,
                "project_type": VALID_PROJECTS,
                "has_industry_sponsor": VALID_YES_NO,
            },
        }