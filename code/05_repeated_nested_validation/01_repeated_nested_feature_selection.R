#!/usr/bin/env Rscript
# ============================================================
# Repeated nested resampling - stage 1: outer-fold assignment
# and BSA-free LASSO/Boruta feature selection.
#
# Design reproduced from the revised manuscript:
# - Full archived analysis-ready cohort: n=3368, events=1357.
# - 5 repeats x stratified 5-fold outer resampling = 25 folds.
# - Within each outer training fold:
#     * exclude BSA before selection;
#     * remove training-fold zero-variance variables;
#     * LASSO: glmnet binomial, alpha=1, 10-fold CV, lambda.1se;
#     * Boruta: 1000 trees, maxRuns=100, TentativeRoughFix;
#     * final set = LASSO intersection Boruta, using union only if
#       the intersection contains fewer than 3 variables.
#
# IMPORTANT LIMITATION:
# The archived revision input is already imputed and winsorized.
# This stage cannot refit the original imputation models or
# winsorization limits inside outer folds. It validates the
# recoverable post-preprocessing pipeline only.
# ============================================================

rm(list = ls())

# --------------------------
# Minimal CLI parser
# --------------------------
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit) == 0) return(default)
  sub(prefix, "", hit[[1]], fixed = TRUE)
}

INPUT_FILE <- get_arg("input", Sys.getenv("AIS_ICU_ANALYSIS_READY", unset = ""))
OUT_DIR <- get_arg("output-dir", "outputs/repeated_nested")

if (INPUT_FILE == "") {
  stop("Provide --input=/path/to/restricted/analysis_ready.csv or set AIS_ICU_ANALYSIS_READY.")
}
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

required_pkgs <- c("glmnet", "Boruta", "caret", "ranger")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing_pkgs) > 0) {
  stop(
    "Missing R packages: ", paste(missing_pkgs, collapse = ", "),
    ". Install the versions listed in the repository environment documentation."
  )
}

suppressPackageStartupMessages({
  library(glmnet)
  library(Boruta)
  library(caret)
  library(ranger)
})

OUTCOME <- "Pulmonary_infection"
ID_COL <- "Study_row_id"
N_REPEATS <- 5
OUTER_FOLDS <- 5
RANDOM_STATE <- 20260901

# Original 38 candidate fields. BSA is excluded a priori in every outer fold.
ALL_CANDIDATES <- c(
  "Age", "Sex", "Admission_type",
  "CCI", "Diabetes_mellitus", "Heart_disease",
  "Surgery", "Mechanical_ventilation", "Intubation_tracheotomy",
  "Central_venous_catheter", "CRRT",
  "Immunosuppressants", "Anticholinergics", "Anticholinesterases",
  "Benzodiazepines", "Non_benzodiazepines", "Antipsychotics",
  "Vasoactive_agents", "Broad_spectrum_antibiotics", "Diuretics",
  "WBC", "NEUT_abs", "LYMPH_abs", "HGB", "PLT",
  "INR", "FIB", "ALB", "CREA", "BUN", "ALT", "HbA1c",
  "NA", "K", "CL", "CO2", "RBC", "LDH"
)

if (length(ALL_CANDIDATES) != 38) stop("Candidate inventory is not 38 fields.")

BSA <- "Broad_spectrum_antibiotics"
BSA_FREE_CANDIDATES <- setdiff(ALL_CANDIDATES, BSA)

BINARY_VARS <- c(
  "Sex", "Surgery", "Mechanical_ventilation", "Intubation_tracheotomy",
  "Central_venous_catheter", "CRRT", "Diabetes_mellitus", "Heart_disease",
  "Immunosuppressants", "Anticholinergics", "Anticholinesterases",
  "Benzodiazepines", "Non_benzodiazepines", "Antipsychotics",
  "Vasoactive_agents", "Diuretics"
)
CATEGORICAL_VARS <- c("Admission_type")
CONTINUOUS_VARS <- setdiff(BSA_FREE_CANDIDATES, c(BINARY_VARS, CATEGORICAL_VARS))

DISPLAY_NAME <- c(
  Age="Age", Sex="Sex", Admission_type="Admission type",
  CCI="CCI", Diabetes_mellitus="DM", Heart_disease="Heart disease",
  Surgery="Surgery", Mechanical_ventilation="MV",
  Intubation_tracheotomy="Intubation/tracheotomy",
  Central_venous_catheter="CVC", CRRT="CRRT",
  Immunosuppressants="Immunosuppressants", Anticholinergics="Anticholinergics",
  Anticholinesterases="Cholinesterase inhibitors", Benzodiazepines="Benzodiazepines",
  Non_benzodiazepines="Non-benzodiazepines", Antipsychotics="Antipsychotics",
  Vasoactive_agents="Vasoactive agents", Diuretics="Diuretics",
  WBC="WBC", NEUT_abs="NEU", LYMPH_abs="LYM", HGB="Hb", PLT="PLT",
  INR="INR", FIB="FIB", ALB="ALB", CREA="SCr", BUN="BUN", ALT="ALT",
  HbA1c="HbA1c", NA="Na", K="K", CL="Cl", CO2="TCO2", RBC="RBC", LDH="LDH"
)

read_csv_safe <- function(path) {
  for (enc in c("UTF-8-BOM", "UTF-8", "GB18030", "GBK", "latin1")) {
    d <- try(read.csv(path, check.names = FALSE, fileEncoding = enc,
                      stringsAsFactors = FALSE), silent = TRUE)
    if (!inherits(d, "try-error")) return(d)
  }
  stop("Unable to read input CSV: ", path)
}

dat <- read_csv_safe(INPUT_FILE)
colnames(dat) <- trimws(sub("^\\ufeff", "", colnames(dat)))

required <- c(ID_COL, OUTCOME, ALL_CANDIDATES)
missing_cols <- setdiff(required, colnames(dat))
if (length(missing_cols) > 0) stop("Missing columns: ", paste(missing_cols, collapse = ", "))
if (nrow(dat) != 3368) stop("Expected n=3368; observed n=", nrow(dat))
if (anyDuplicated(dat[[ID_COL]]) > 0) stop("Duplicate Study_row_id detected.")

# Normalize binary outcome to integer 0/1.
y <- suppressWarnings(as.integer(as.character(dat[[OUTCOME]])))
if (any(is.na(y)) || !all(y %in% c(0L, 1L))) stop("Outcome must be binary 0/1.")
dat[[OUTCOME]] <- y
if (sum(y) != 1357) stop("Expected 1357 events; observed ", sum(y))

# Archived analysis-ready input should be complete for candidate fields.
if (sum(is.na(dat[, BSA_FREE_CANDIDATES, drop=FALSE])) > 0) {
  stop("Unexpected missing values in archived analysis-ready candidate fields.")
}

# Factor treatment mirrors the fixed-training primary selection code.
prepare_selection_frame <- function(d, predictors) {
  out <- d[, c(predictors, OUTCOME), drop = FALSE]
  for (v in intersect(BINARY_VARS, predictors)) {
    lev <- sort(unique(out[[v]]))
    out[[v]] <- factor(out[[v]], levels = lev)
  }
  for (v in intersect(CATEGORICAL_VARS, predictors)) {
    lev <- sort(unique(out[[v]]))
    out[[v]] <- factor(out[[v]], levels = lev)
  }
  for (v in intersect(CONTINUOUS_VARS, predictors)) {
    out[[v]] <- suppressWarnings(as.numeric(out[[v]]))
  }
  out[[OUTCOME]] <- factor(out[[OUTCOME]], levels = c(0, 1), labels = c("No", "Yes"))
  out
}

map_dummy_to_var <- function(dummy, vars) {
  if (dummy %in% vars) return(dummy)
  hits <- vars[vapply(vars, function(v) startsWith(dummy, v), logical(1))]
  if (length(hits) == 0) return(dummy)
  hits[which.max(nchar(hits))]
}

# ------------------------------------------------------------
# Outer folds: caret createMultiFolds, set once for exact reruns.
# createMultiFolds returns training indices; complement = validation.
# ------------------------------------------------------------
set.seed(RANDOM_STATE)
outer_train_sets <- createMultiFolds(
  factor(dat[[OUTCOME]], levels=c(0,1)),
  k = OUTER_FOLDS,
  times = N_REPEATS
)

# Parse names such as Fold1.Rep1.
parse_fold_name <- function(nm) {
  m <- regexec("Fold([0-9]+)\\.Rep([0-9]+)", nm)
  z <- regmatches(nm, m)[[1]]
  if (length(z) != 3) stop("Unexpected caret fold name: ", nm)
  c(Fold=as.integer(z[2]), Repeat=as.integer(z[3]))
}

assignment_rows <- list()
selection_rows <- list()
lasso_count <- setNames(rep(0L, length(BSA_FREE_CANDIDATES)), BSA_FREE_CANDIDATES)
boruta_count <- setNames(rep(0L, length(BSA_FREE_CANDIDATES)), BSA_FREE_CANDIDATES)
final_count <- setNames(rep(0L, length(BSA_FREE_CANDIDATES)), BSA_FREE_CANDIDATES)

for (nm in names(outer_train_sets)) {
  rf <- parse_fold_name(nm)
  f <- unname(rf["Fold"])
  r <- unname(rf["Repeat"])
  tr_idx <- outer_train_sets[[nm]]
  va_idx <- setdiff(seq_len(nrow(dat)), tr_idx)

  assignment_rows[[length(assignment_rows) + 1]] <- data.frame(
    Study_row_id = dat[[ID_COL]][va_idx],
    Repeat = r,
    Fold = f,
    stringsAsFactors = FALSE
  )

  dtr <- dat[tr_idx, , drop=FALSE]

  # BSA is already excluded. Remove any training-fold zero-variance fields.
  zero_vars <- BSA_FREE_CANDIDATES[vapply(
    BSA_FREE_CANDIDATES,
    function(v) length(unique(dtr[[v]][!is.na(dtr[[v]])])) <= 1,
    logical(1)
  )]
  predictors <- setdiff(BSA_FREE_CANDIDATES, zero_vars)
  if (length(predictors) < 3) stop("Too few non-zero-variance predictors in outer fold.")

  train_complete <- prepare_selection_frame(dtr, predictors)

  # ----- LASSO: 10-fold CV, lambda.1se -----
  x <- model.matrix(
    as.formula(paste0("`", OUTCOME, "` ~ .")),
    data = train_complete
  )[, -1, drop=FALSE]
  yy <- as.numeric(train_complete[[OUTCOME]]) - 1

  fold_seed <- RANDOM_STATE + r * 100 + f
  set.seed(fold_seed)
  cvfit <- cv.glmnet(
    x, yy, family="binomial", alpha=1, nfolds=10, maxit=1e5
  )
  cm <- coef(cvfit, s="lambda.1se")
  nz_terms <- rownames(cm)[as.numeric(cm) != 0]
  nz_terms <- setdiff(nz_terms, "(Intercept)")
  lasso_vars <- unique(vapply(nz_terms, map_dummy_to_var, character(1), vars=predictors))
  lasso_vars <- intersect(predictors, lasso_vars)

  # ----- Boruta: 1000 trees, maxRuns=100, TentativeRoughFix -----
  set.seed(fold_seed + 50000)
  bout <- Boruta(
    as.formula(paste0("`", OUTCOME, "` ~ .")),
    data = train_complete,
    doTrace = 0,
    maxRuns = 100,
    num.trees = 1000
  )
  bfix <- TentativeRoughFix(bout)
  boruta_vars <- getSelectedAttributes(bfix, withTentative=FALSE)
  boruta_vars <- intersect(predictors, boruta_vars)

  common_vars <- intersect(lasso_vars, boruta_vars)
  union_vars <- union(lasso_vars, boruta_vars)
  if (length(common_vars) >= 3) {
    final_vars <- common_vars
    final_rule <- "intersection"
  } else {
    final_vars <- union_vars
    final_rule <- "union_because_intersection_lt3"
  }

  # Keep Python-consumable raw column names in the stage-1 output.
  lasso_count[lasso_vars] <- lasso_count[lasso_vars] + 1L
  boruta_count[boruta_vars] <- boruta_count[boruta_vars] + 1L
  final_count[final_vars] <- final_count[final_vars] + 1L

  selection_rows[[length(selection_rows) + 1]] <- data.frame(
    Repeat = r,
    Fold = f,
    N_train = length(tr_idx),
    Events_train = sum(dat[[OUTCOME]][tr_idx]),
    N_validation = length(va_idx),
    Events_validation = sum(dat[[OUTCOME]][va_idx]),
    N_candidate_BSA_free = length(BSA_FREE_CANDIDATES),
    N_zero_variance = length(zero_vars),
    Zero_variance_vars = paste(zero_vars, collapse=";"),
    N_entered_selection = length(predictors),
    lambda_min = cvfit$lambda.min,
    lambda_1se = cvfit$lambda.1se,
    N_LASSO = length(lasso_vars),
    LASSO_vars = paste(lasso_vars, collapse=";"),
    N_Boruta = length(boruta_vars),
    Boruta_vars = paste(boruta_vars, collapse=";"),
    N_Final = length(final_vars),
    Final_vars = paste(final_vars, collapse=";"),
    Final_rule = final_rule,
    stringsAsFactors = FALSE
  )

  cat(sprintf(
    "Repeat %d Fold %d | train=%d val=%d | zero=%d | LASSO=%d Boruta=%d Final=%d\n",
    r, f, length(tr_idx), length(va_idx), length(zero_vars),
    length(lasso_vars), length(boruta_vars), length(final_vars)
  ))
}

assign_df <- do.call(rbind, assignment_rows)
sel_df <- do.call(rbind, selection_rows)
assign_df <- assign_df[order(assign_df$Repeat, assign_df$Fold, assign_df$Study_row_id), ]
sel_df <- sel_df[order(sel_df$Repeat, sel_df$Fold), ]

# Integrity checks: 25 folds; each patient appears once per repeat as validation.
if (nrow(sel_df) != N_REPEATS * OUTER_FOLDS) stop("Stage-1 did not create 25 outer folds.")
per_patient <- table(assign_df$Study_row_id)
if (length(per_patient) != nrow(dat) || any(per_patient != N_REPEATS)) {
  stop("Outer assignment integrity failure: each patient must appear once per repeat in validation.")
}

write.csv(assign_df, file.path(OUT_DIR, "00_outer_fold_assignments.csv"), row.names=FALSE)
write.csv(sel_df, file.path(OUT_DIR, "01_nested_outer_fold_feature_selection.csv"), row.names=FALSE)

freq_df <- data.frame(
  Variable_raw = BSA_FREE_CANDIDATES,
  Variable = unname(DISPLAY_NAME[BSA_FREE_CANDIDATES]),
  LASSO_selected_n = as.integer(lasso_count[BSA_FREE_CANDIDATES]),
  Boruta_confirmed_n = as.integer(boruta_count[BSA_FREE_CANDIDATES]),
  Final_selected_n = as.integer(final_count[BSA_FREE_CANDIDATES]),
  Total_outer_folds = N_REPEATS * OUTER_FOLDS,
  stringsAsFactors = FALSE
)
freq_df$LASSO_selected_pct <- 100 * freq_df$LASSO_selected_n / freq_df$Total_outer_folds
freq_df$Boruta_confirmed_pct <- 100 * freq_df$Boruta_confirmed_n / freq_df$Total_outer_folds
freq_df$Final_selected_pct <- 100 * freq_df$Final_selected_n / freq_df$Total_outer_folds
freq_df <- freq_df[order(-freq_df$Final_selected_n, -freq_df$LASSO_selected_n, freq_df$Variable), ]
write.csv(freq_df, file.path(OUT_DIR, "02_nested_feature_selection_frequency.csv"), row.names=FALSE)

readme <- c(
  "Repeated nested resampling stage 1 - feature selection",
  "====================================================",
  paste0("Input: ", INPUT_FILE),
  paste0("Cohort: n=", nrow(dat), "; events=", sum(dat[[OUTCOME]])),
  paste0("Outer design: ", N_REPEATS, " repeats x ", OUTER_FOLDS, " folds = ", nrow(sel_df), " validation folds"),
  paste0("Outer assignment seed: ", RANDOM_STATE),
  "",
  "Within each outer training fold:",
  "- broad-spectrum antibiotics were excluded before selection;",
  "- training-fold zero-variance fields were removed;",
  "- LASSO used binomial glmnet, alpha=1, 10-fold CV, lambda.1se;",
  "- Boruta used 1000 trees, maxRuns=100, followed by TentativeRoughFix;",
  "- final selection used the LASSO-Boruta intersection, with union only when intersection <3 variables.",
  "",
  "Critical limitation:",
  "The archived revision input had already undergone the original imputation and winsorization workflow.",
  "Those stages cannot be re-estimated within outer folds and are not claimed as part of this repeated nested validation."
)
writeLines(readme, file.path(OUT_DIR, "README_stage1.txt"), useBytes=TRUE)

cat("\nDONE\n")
cat("Saved:\n")
cat("  00_outer_fold_assignments.csv\n")
cat("  01_nested_outer_fold_feature_selection.csv\n")
cat("  02_nested_feature_selection_frequency.csv\n")
