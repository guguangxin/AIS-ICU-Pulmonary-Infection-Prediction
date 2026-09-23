# ============================================================
# ICU 脑卒中人群肺部感染预测模型 —— 主分析候选变量重筛
# 结局：Pulmonary_infection (肺部感染)
#
# ★ 完全沿用原变量筛选代码的核心逻辑：
#   候选变量直接进入 LASSO + Boruta
#   不做相关性预筛选
#   LASSO: 10折CV + lambda.1se
#   Boruta: 1000棵树 + 100轮 + TentativeRoughFix
#   汇总交集 / 并集 / 最终推荐
#
# ★ 本次把审稿意见正式落实到主分析变量筛选流程：
#   1) 在任何数据驱动筛选之前，先验排除 BSA
#   2) 因全队列 Immunosuppressants = 0/3368，作为无信息零方差变量先验排除
#   3) 因 standalone Heart disease 字段经数据质量审计确认不能代表完整心脏共病，作为 data-quality exclusion 在筛选前排除
#   4) 因此 38 个原始候选 -> 35 个先验/数据质量排除后候选
#   5) 在固定主分析训练集内再做技术性零方差检查；零方差变量不进入 LASSO/Boruta
#      （按当前数据，Cholinesterase inhibitors 在训练集为全0，因此预计 34 个变量实际进入算法）
#   6) 使用主分析已经固定的原始 Train/Test 身份，
#      不再用 caret::createDataPartition 重新随机拆分
#   7) 单独输出 Age 的 LASSO 系数和 Boruta Z-score/decision
#   8) 增加20次重复稳定性分析：
#      始终使用同一个固定训练集，不重新抽患者，不重新划分Train/Test；
#      LASSO仅改变CV随机折分，Boruta仅改变随机种子
# ============================================================

rm(list = ls())

# ── 路径 ──────────────────────────────────────────────────
INPUT_FILE <- r"(E:\新建文件夹\第三次修稿\BSAfree_feature_selection_input_exactsplit.csv)"
SAVE_PATH  <- r"(E:\新建文件夹\第三次修稿\主分析_HeartDisease数据质量排除_34候选重筛)"
if (!dir.exists(SAVE_PATH)) dir.create(SAVE_PATH, recursive = TRUE)
setwd(SAVE_PATH)

# ── 包 ────────────────────────────────────────────────────
for (pkg in c("glmnet", "Boruta", "caret", "pROC")) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE))
    install.packages(pkg, repos = "https://cran.r-project.org")
  library(pkg, character.only = TRUE)
}

# ── 参数 ──────────────────────────────────────────────────
OUTCOME        <- "Pulmonary_infection"
OUTCOME_CN     <- "肺部感染_主分析重筛"
COR_OUTCOME_TH <- 0.05       # 仅用于Spearman条形图参考线，不用于筛选

LASSO_SEED     <- 42
BORUTA_SEED    <- 123

RUN_STABILITY  <- TRUE       # TRUE = 再做20次LASSO/Boruta重复稳定性
N_REPEATS      <- 20

# ── 变量名映射（原始列名 → 标准缩写） ──────────────────────
NAME_MAP <- c(
  # 人口学
  "Age"                        = "Age",
  "Sex"                        = "Sex",
  "Admission_type"             = "Admission type",

  # 共病
  "CCI"                        = "CCI",
  "Diabetes_mellitus"          = "DM",
  "Heart_disease"              = "Heart disease",

  # 生命支持
  "Surgery"                    = "Surgery",
  "Mechanical_ventilation"     = "MV",
  "Intubation_tracheotomy"     = "Intubation",
  "Central_venous_catheter"    = "CVC",
  "CRRT"                       = "CRRT",

  # 用药
  "Immunosuppressants"         = "Immunosuppressants",
  "Anticholinergics"           = "Anticholinergics",
  "Anticholinesterases"        = "Cholinesterase inhibitors",
  "Benzodiazepines"            = "Benzodiazepines",
  "Non_benzodiazepines"        = "Non-benzodiazepines",
  "Antipsychotics"             = "Antipsychotics",
  "Vasoactive_agents"          = "Vasoactive agents",
  "Broad_spectrum_antibiotics" = "BSA",
  "Diuretics"                  = "Diuretics",

  # 血常规
  "WBC"                        = "WBC",
  "NEUT_abs"                   = "NEU",
  "LYMPH_abs"                  = "LYM",
  "HGB"                        = "Hb",
  "PLT"                        = "PLT",

  # 凝血
  "INR"                        = "INR",
  "FIB"                        = "FIB",

  # 肝肾代谢
  "ALB"                        = "ALB",
  "CREA"                       = "CREA",
  "BUN"                        = "BUN",
  "ALT"                        = "ALT",
  "HbA1c"                      = "HbA1c",

  # 电解质 / 血气
  "NA"                         = "Na",
  "K"                          = "K",
  "CL"                         = "Cl",
  "CO2"                        = "TCO2",
  "RBC"                        = "RBC",

  # 其他
  "LDH"                        = "LDH"
)

BINARY_VARS_ALL <- unname(NAME_MAP[c(
  "Sex",
  "Surgery", "Mechanical_ventilation", "Intubation_tracheotomy",
  "Central_venous_catheter", "CRRT",
  "Diabetes_mellitus", "Heart_disease",
  "Immunosuppressants", "Anticholinergics", "Anticholinesterases",
  "Benzodiazepines", "Non_benzodiazepines", "Antipsychotics",
  "Vasoactive_agents", "Broad_spectrum_antibiotics", "Diuretics"
)])

CATEGORICAL_VARS_ALL <- unname(NAME_MAP["Admission_type"])

CONTINUOUS_VARS_ALL <- unname(NAME_MAP[c(
  "Age", "CCI",
  "WBC", "NEUT_abs", "LYMPH_abs", "HGB", "PLT",
  "INR", "FIB",
  "ALB", "CREA", "BUN", "ALT", "HbA1c",
  "NA", "K", "CL", "CO2", "RBC",
  "LDH"
)])

ALL_PREDICTORS <- c(
  BINARY_VARS_ALL,
  CATEGORICAL_VARS_ALL,
  CONTINUOUS_VARS_ALL
)

# ★ 主分析先验排除规则
# BSA：因其时间相对感染怀疑存在潜在内生性/信息泄漏风险，筛选前排除
# Immunosuppressants：全研究队列为0/3368，无任何变异与预测信息，筛选前排除
# Heart disease：standalone结构化字段经审计仅9例阳性，不能代表完整心脏共病；与CCI中基于ICD-10的MI/CHF等心脏组件不同，按数据质量原因排除
A_PRIORI_EXCLUDED <- c("BSA", "Immunosuppressants", "Heart disease")
PREDICTORS_AFTER_EXCLUSION <- setdiff(ALL_PREDICTORS, A_PRIORI_EXCLUDED)

# 后面训练集零方差核查后，真正送入LASSO/Boruta的是 PREDICTORS
PREDICTORS <- PREDICTORS_AFTER_EXCLUSION

cat("原候选预测变量数:", length(ALL_PREDICTORS), "\n")
cat("先验排除:", paste(A_PRIORI_EXCLUDED, collapse = ", "), "\n")
cat("先验排除后候选变量数:", length(PREDICTORS_AFTER_EXCLUSION), "\n")

if (length(ALL_PREDICTORS) != 38) {
  stop("原候选变量数不是38，请检查变量映射。")
}
if (length(PREDICTORS_AFTER_EXCLUSION) != 35) {
  stop("删除BSA、Immunosuppressants和standalone Heart disease后不是35个候选变量，请检查。")
}

GROUP_MAP <- list(
  Demographics  = unname(NAME_MAP[c("Age", "Sex", "Admission_type")]),
  Comorbidities = unname(NAME_MAP[c("CCI", "Diabetes_mellitus", "Heart_disease")]),
  `Life Support`= unname(NAME_MAP[c(
    "Surgery", "Mechanical_ventilation",
    "Intubation_tracheotomy",
    "Central_venous_catheter", "CRRT"
  )]),
  Medications   = unname(NAME_MAP[c(
    "Immunosuppressants", "Anticholinergics",
    "Anticholinesterases",
    "Benzodiazepines", "Non_benzodiazepines",
    "Antipsychotics", "Vasoactive_agents",
    "Broad_spectrum_antibiotics", "Diuretics"
  )])
)

get_group <- function(v) {
  for (g in names(GROUP_MAP)) {
    if (v %in% GROUP_MAP[[g]]) return(g)
  }
  return("Lab Tests")
}


# ============================================================
# 1. 读取数据
# ============================================================
cat("\n========== 1. 读取数据 ==========\n")

read_csv_safe <- function(path) {
  for (enc in c("UTF-8-BOM", "UTF-8", "GBK", "GB18030", "latin1")) {
    dat <- try(
      read.csv(
        path,
        check.names = FALSE,
        fileEncoding = enc,
        stringsAsFactors = FALSE
      ),
      silent = TRUE
    )
    if (!inherits(dat, "try-error")) {
      cat(
        "数据读取成功 (编码 =", enc, "):",
        nrow(dat), "行 ×", ncol(dat), "列\n"
      )
      return(dat)
    }
  }
  stop("无法读取文件: ", path)
}

dat <- read_csv_safe(INPUT_FILE)

# 本次输入表必须带主分析固定划分身份
required_meta <- c("Study_row_id", "Primary_split", OUTCOME)
miss_meta <- setdiff(required_meta, colnames(dat))
if (length(miss_meta) > 0) {
  stop(
    "输入文件缺少以下主分析身份列: ",
    paste(miss_meta, collapse = ", ")
  )
}

miss_cols <- setdiff(c(names(NAME_MAP), OUTCOME), colnames(dat))
if (length(miss_cols) > 0) {
  cat("⚠️ 以下列在数据中不存在，请检查列名：\n")
  print(miss_cols)
  stop("请确认列名后重新运行")
}

# ── 执行列名重命名 ────────────────────────────────────────
for (old_name in names(NAME_MAP)) {
  new_name <- NAME_MAP[[old_name]]
  if (old_name != new_name) {
    colnames(dat)[colnames(dat) == old_name] <- new_name
  }
}

cat("列名已按标准缩写重命名\n")

# ── 类型转换 ──────────────────────────────────────────────
BINARY_VARS <- intersect(BINARY_VARS_ALL, colnames(dat))
CATEGORICAL_VARS <- intersect(CATEGORICAL_VARS_ALL, colnames(dat))
CONTINUOUS_VARS <- intersect(CONTINUOUS_VARS_ALL, colnames(dat))

for (v in BINARY_VARS) {
  # 保留原始实际编码水平。特别是 Sex 在本数据中编码为 1/2，
  # 不能强制 levels=c(0,1)，否则所有“2”会被转成 NA，造成伪零方差。
  lev <- sort(unique(na.omit(dat[[v]])))
  if (length(lev) > 2) {
    stop(paste0("二元变量 ", v, " 实际包含超过2个水平: ",
                paste(lev, collapse = ", ")))
  }
  dat[[v]] <- factor(dat[[v]], levels = lev)
}

for (v in CATEGORICAL_VARS) {
  lev <- sort(unique(na.omit(dat[[v]])))
  dat[[v]] <- factor(dat[[v]], levels = lev)
}

for (v in CONTINUOUS_VARS) {
  dat[[v]] <- suppressWarnings(as.numeric(dat[[v]]))
}

dat[[OUTCOME]] <- factor(
  dat[[OUTCOME]],
  levels = c(0, 1),
  labels = c("No", "Yes")
)

# ============================================================
# 2. 使用主分析固定 Train / Test 身份
# ============================================================
cat("\n========== 2. 固定主分析 Train / Test ==========\n")

if (!all(dat$Primary_split %in% c("Train", "Test"))) {
  stop("Primary_split 只能包含 Train / Test。")
}

train_set <- dat[dat$Primary_split == "Train", ]
test_set  <- dat[dat$Primary_split == "Test", ]

cat(
  "训练集:", nrow(train_set), "行 |",
  "阳性:", sum(train_set[[OUTCOME]] == "Yes"), "\n"
)
cat(
  "测试集:", nrow(test_set), "行 |",
  "阳性:", sum(test_set[[OUTCOME]] == "Yes"), "\n"
)

# 必须精确复现现行主分析
if (
  nrow(train_set) != 2357 ||
  sum(train_set[[OUTCOME]] == "Yes") != 950
) {
  stop("训练集未复现为 n=2357 / 阳性=950，请停止。")
}

if (
  nrow(test_set) != 1011 ||
  sum(test_set[[OUTCOME]] == "Yes") != 407
) {
  stop("测试集未复现为 n=1011 / 阳性=407，请停止。")
}

write.csv(
  data.frame(
    Set = c("Train", "Test"),
    N = c(nrow(train_set), nrow(test_set)),
    Positive = c(
      sum(train_set[[OUTCOME]] == "Yes"),
      sum(test_set[[OUTCOME]] == "Yes")
    )
  ),
  "00_主分析固定TrainTest核查.csv",
  row.names = FALSE
)


# ============================================================
# 3. 先验排除 + 训练集零方差核查
# ============================================================
cat("\n========== 3. 先验排除 + 训练集零方差核查 ==========\n")

write.csv(
  data.frame(
    Variable = c("BSA", "Immunosuppressants", "Heart disease"),
    Exclusion_stage = c("A priori before feature selection", "A priori before feature selection", "Data-quality exclusion before feature selection"),
    Reason = c(
      "Potentially endogenous first-48-hour treatment exposure; timing relative to emerging infection suspicion cannot be reconstructed reliably",
      "Zero variance in the entire cohort (0/3368 exposed); no information for feature selection",
      "Known non-functional standalone structured field (9/3368 positive) that does not validly represent overall cardiac comorbidity; distinct from ICD-10-derived Charlson cardiac components"
    ),
    stringsAsFactors = FALSE
  ),
  "00_主分析先验排除变量与原因.csv",
  row.names = FALSE
)

candidate_screen <- data.frame(
  Variable = PREDICTORS_AFTER_EXCLUSION,
  Group = sapply(PREDICTORS_AFTER_EXCLUSION, get_group),
  N_unique_train = NA_integer_,
  Missing_train = NA_integer_,
  Zero_variance = FALSE,
  Minority_n = NA_integer_,
  Minority_pct = NA_real_,
  stringsAsFactors = FALSE
)

for (i in seq_along(PREDICTORS_AFTER_EXCLUSION)) {
  v <- PREDICTORS_AFTER_EXCLUSION[i]
  x <- train_set[[v]]

  observed <- x[!is.na(x)]
  n_unique <- length(unique(observed))

  candidate_screen$N_unique_train[i] <- n_unique
  candidate_screen$Missing_train[i] <- sum(is.na(x))
  candidate_screen$Zero_variance[i] <- (n_unique <= 1)

  # 对二元变量记录少数类人数/比例
  if (v %in% BINARY_VARS) {
    tb <- table(x, useNA = "no")
    if (length(tb) >= 2) {
      candidate_screen$Minority_n[i] <- min(tb)
      candidate_screen$Minority_pct[i] <- 100 * min(tb) / sum(tb)
    } else if (length(tb) == 1) {
      candidate_screen$Minority_n[i] <- 0
      candidate_screen$Minority_pct[i] <- 0
    }
  }
}

candidate_screen$Rare_binary_lt10 <- (
  !is.na(candidate_screen$Minority_n) &
  candidate_screen$Minority_n < 10
)

write.csv(
  candidate_screen,
  "01_主分析35候选变量_零方差与稀有度核查.csv",
  row.names = FALSE
)

zero_vars <- candidate_screen$Variable[
  candidate_screen$Zero_variance
]

if (length(zero_vars) > 0) {
  cat(
    "训练集零方差变量（本次不进入LASSO/Boruta）:",
    paste(zero_vars, collapse = ", "), "\n"
  )
}

rare_vars <- candidate_screen$Variable[
  candidate_screen$Rare_binary_lt10 &
  !candidate_screen$Zero_variance
]

if (length(rare_vars) > 0) {
  cat(
    "训练集极稀有二元变量（保留，但单独报告）:",
    paste(rare_vars, collapse = ", "), "\n"
  )
}

# ★ 真正进入LASSO/Boruta的变量
PREDICTORS <- setdiff(PREDICTORS_AFTER_EXCLUSION, zero_vars)

if (any(A_PRIORI_EXCLUDED %in% PREDICTORS)) {
  stop("先验排除变量意外进入了LASSO/Boruta，请停止检查。")
}

cat("先验排除后变量数:", length(PREDICTORS_AFTER_EXCLUSION), "\n")
cat("零方差剔除后实际进入筛选:", length(PREDICTORS), "\n")
if (length(PREDICTORS) != 34) {
  stop(paste0("预期34个变量进入LASSO/Boruta，实际=", length(PREDICTORS), "。请检查。"))
}

# 注意：当前input是analysis-ready数据；若仍有缺失，只使用完整案例，
# 保持与你原代码一致，不在此重新发明新的插补流程。
dat_use <- dat[, c("Primary_split", OUTCOME, PREDICTORS), drop = FALSE]

train_complete <- train_set[, c(OUTCOME, PREDICTORS), drop = FALSE]
train_complete <- train_complete[complete.cases(train_complete), ]

test_complete <- test_set[, c(OUTCOME, PREDICTORS), drop = FALSE]
test_complete <- test_complete[complete.cases(test_complete), ]

cat(
  "训练集完整案例数:", nrow(train_complete), "行 ×",
  ncol(train_complete), "列\n"
)
cat(
  "测试集完整案例数:", nrow(test_complete), "行 ×",
  ncol(test_complete), "列\n"
)

n_pos <- sum(train_complete[[OUTCOME]] == "Yes")
cat("训练集阳性事件数 =", n_pos, "\n")
cat(
  "EPV提示：建议最终入模变量数 ≤",
  floor(n_pos / 10), "个 (仅作描述)\n"
)


# ============================================================
# 4. 与结局 Spearman 相关性
#    仅用于报告/可视化，不用于筛选
# ============================================================
cat(
  "\n========== 4. 与结局 Spearman 相关性",
  " (信息性指标，不筛选) ==========\n"
)

to_numeric_safe <- function(x) {
  if (is.factor(x)) as.numeric(as.character(x))
  else if (is.numeric(x)) x
  else suppressWarnings(as.numeric(x))
}

y_train_num <- as.numeric(train_complete[[OUTCOME]]) - 1

cor_with_outcome <- function(df, vars, y_num) {
  res <- data.frame(
    Variable = vars,
    Type = NA_character_,
    Spearman_r = NA_real_,
    Abs_r = NA_real_,
    P_value = NA_real_,
    stringsAsFactors = FALSE
  )

  for (i in seq_along(vars)) {
    v <- vars[i]
    x <- df[[v]]

    is_cat <- (
      is.factor(x) ||
      v %in% c(BINARY_VARS, CATEGORICAL_VARS)
    )

    res$Type[i] <- ifelse(
      is_cat,
      "Categorical",
      "Continuous"
    )

    x_num <- to_numeric_safe(x)
    ok <- !is.na(x_num) & !is.na(y_num)

    if (sum(ok) < 10 || sd(x_num[ok]) == 0) {
      res$Spearman_r[i] <- 0
      res$P_value[i] <- 1
    } else {
      ct <- suppressWarnings(
        cor.test(
          x_num[ok],
          y_num[ok],
          method = "spearman",
          exact = FALSE
        )
      )
      res$Spearman_r[i] <- as.numeric(ct$estimate)
      res$P_value[i] <- ct$p.value
    }

    res$Abs_r[i] <- abs(res$Spearman_r[i])
  }

  res <- res[order(res$Abs_r, decreasing = TRUE), ]
  list(df = res)
}

scr <- cor_with_outcome(
  train_complete,
  PREDICTORS,
  y_train_num
)

write.csv(
  scr$df,
  paste0(OUTCOME_CN, "_与结局Spearman相关.csv"),
  row.names = FALSE
)

cat(
  "★ Spearman只用于报告；全部",
  length(PREDICTORS),
  "个非零方差主分析候选变量直接进入LASSO和Boruta\n"
)


# ============================================================
# 5. LASSO（10折CV）
# ============================================================
cat("\n========== 5. LASSO (10 折 CV) ==========\n")

lasso_x <- model.matrix(
  as.formula(paste0("`", OUTCOME, "` ~ .")),
  data = train_complete
)[, -1, drop = FALSE]

lasso_y <- as.numeric(train_complete[[OUTCOME]]) - 1

set.seed(LASSO_SEED)
fit_path <- glmnet(
  lasso_x,
  lasso_y,
  family = "binomial",
  alpha = 1,
  maxit = 1e5
)

# ---- LASSO作图统一使用 log(lambda) 横轴（不用 -log(lambda)） ----
# 这样与glmnet经典展示和当前稿件Figure 1保持同一方向：
# 向右 = lambda增大 = 惩罚增强 = 非零系数减少/系数收缩至0。
plot_glmnet_path_loglambda <- function(fit, lwd = 1.8) {
  x <- log(fit$lambda)
  beta <- as.matrix(fit$beta)

  # glmnet内部lambda通常按从大到小保存；作图前按log(lambda)升序排列，
  # 使横轴自然从左到右递增。
  ord <- order(x)
  x <- x[ord]
  beta <- beta[, ord, drop = FALSE]
  df <- fit$df[ord]

  matplot(
    x,
    t(beta),
    type = "l",
    lty = 1,
    lwd = lwd,
    xlab = expression(log(lambda)),
    ylab = "Coefficients",
    main = ""
  )

  # 顶部显示对应lambda处的非零系数个数，方向与横轴一致。
  tick_idx <- unique(round(seq(1, length(x), length.out = min(7, length(x)))))
  axis(
    3,
    at = x[tick_idx],
    labels = df[tick_idx],
    tick = FALSE,
    line = 0
  )
}

plot_cv_glmnet_loglambda <- function(cvfit, lwd = 2) {
  x <- log(cvfit$lambda)
  ord <- order(x)

  x <- x[ord]
  cvm <- cvfit$cvm[ord]
  cvlo <- cvfit$cvlo[ord]
  cvup <- cvfit$cvup[ord]
  nzero <- cvfit$nzero[ord]

  plot(
    x,
    cvm,
    type = "n",
    xlab = expression(log(lambda)),
    ylab = "Binomial Deviance",
    ylim = range(c(cvlo, cvup), finite = TRUE),
    main = ""
  )

  # 交叉验证误差条和均值点。
  arrows(
    x, cvlo,
    x, cvup,
    angle = 90,
    code = 3,
    length = 0.025,
    col = "grey70",
    lwd = 1
  )
  points(x, cvm, pch = 20, col = "red", cex = 0.9)

  # 顶部显示非零系数个数；从左到右应总体减少。
  tick_idx <- unique(round(seq(1, length(x), length.out = min(11, length(x)))))
  axis(
    3,
    at = x[tick_idx],
    labels = nzero[tick_idx],
    tick = FALSE,
    line = 0
  )

  # lambda.min通常位于lambda.1se左侧，因为lambda.min < lambda.1se。
  abline(
    v = log(cvfit$lambda.min),
    lty = "dashed",
    lwd = lwd,
    col = "blue"
  )
  abline(
    v = log(cvfit$lambda.1se),
    lty = "dashed",
    lwd = lwd,
    col = "red"
  )

  legend(
    "topleft",
    legend = c("lambda.min", "lambda.1se"),
    col = c("blue", "red"),
    lty = "dashed",
    lwd = lwd,
    cex = 1.2,
    bg = "white"
  )
}

png(
  paste0(OUTCOME_CN, "_LASSO路径图.png"),
  width = 2400,
  height = 1800,
  res = 300
)
par(
  mar = c(5.5, 6, 4.5, 2),
  cex.axis = 1.5,
  cex.lab = 1.8
)
plot_glmnet_path_loglambda(fit_path, lwd = 1.8)
dev.off()

set.seed(LASSO_SEED)
lasso_cv <- cv.glmnet(
  lasso_x,
  lasso_y,
  family = "binomial",
  alpha = 1,
  nfolds = 10,
  maxit = 1e5
)

png(
  paste0(OUTCOME_CN, "_LASSO交叉验证.png"),
  width = 2200,
  height = 1800,
  res = 300
)
par(
  mar = c(5.5, 6, 4, 2),
  cex.axis = 1.5,
  cex.lab = 1.8
)
plot_cv_glmnet_loglambda(lasso_cv, lwd = 2)
dev.off()

cat(
  "lambda.min:", round(lasso_cv$lambda.min, 6),
  " | lambda.1se:", round(lasso_cv$lambda.1se, 6),
  "\n"
)

coef_mat <- coef(lasso_cv, s = "lambda.1se")

# ★ 保存全部lambda.1se系数，Age即使为0也能看到
coef_all_df <- data.frame(
  term = rownames(coef_mat),
  coefficient = as.numeric(coef_mat),
  stringsAsFactors = FALSE
)
write.csv(
  coef_all_df,
  paste0(OUTCOME_CN, "_LASSO_lambda1se全部系数.csv"),
  row.names = FALSE
)

coef_df <- subset(
  coef_all_df,
  coefficient != 0 & term != "(Intercept)"
)
coef_df <- coef_df[
  order(abs(coef_df$coefficient), decreasing = TRUE),
]

write.csv(
  coef_df,
  paste0(OUTCOME_CN, "_LASSO系数.csv"),
  row.names = FALSE
)

map_dummy_to_var <- function(dummy, vars) {
  if (dummy %in% vars) return(dummy)

  hits <- vars[
    vapply(
      vars,
      function(p) startsWith(dummy, p),
      logical(1)
    )
  ]

  if (length(hits) == 0) return(dummy)
  hits[which.max(nchar(hits))]
}

lasso_vars <- unique(
  vapply(
    coef_df$term,
    map_dummy_to_var,
    character(1),
    vars = PREDICTORS
  )
)

cat(
  "LASSO选中 (", length(lasso_vars), "): ",
  paste(lasso_vars, collapse = ", "),
  "\n",
  sep = ""
)


# ============================================================
# 6. Boruta（1000棵 / 100轮）
# ============================================================
cat("\n========== 6. Boruta ==========\n")

set.seed(BORUTA_SEED)
boruta_out <- Boruta(
  as.formula(paste0("`", OUTCOME, "` ~ .")),
  data = train_complete,
  doTrace = 1,
  maxRuns = 100,
  num.trees = 1000
)

png(
  paste0(OUTCOME_CN, "_Boruta图.png"),
  width = 4200,
  height = 3200,
  res = 300
)
par(
  mar = c(14, 6, 2, 2),
  cex.axis = 1.2,
  cex.lab = 1.8
)
plot(
  boruta_out,
  las = 2,
  cex.axis = 1.2,
  ylab = "Importance (Z-score)",
  xlab = "",
  main = ""
)
dev.off()

# ★ 与原代码相同：TentativeRoughFix 后取明确选中变量
boruta_fixed <- TentativeRoughFix(boruta_out)

boruta_vars <- getSelectedAttributes(
  boruta_fixed,
  withTentative = FALSE
)

boruta_imp <- attStats(boruta_fixed)
boruta_imp$variable <- rownames(boruta_imp)
boruta_imp <- boruta_imp[
  order(boruta_imp$meanImp, decreasing = TRUE),
]

write.csv(
  boruta_imp,
  paste0(OUTCOME_CN, "_Boruta重要性.csv"),
  row.names = FALSE
)

cat(
  "Boruta选中 (", length(boruta_vars), "): ",
  paste(boruta_vars, collapse = ", "),
  "\n",
  sep = ""
)


# ============================================================
# 7. 汇总：交集 / 并集 / 旧9变量比较
# ============================================================
cat("\n========== 7. 结果汇总 ==========\n")

common_vars <- intersect(lasso_vars, boruta_vars)
union_vars  <- union(lasso_vars, boruta_vars)
only_lasso  <- setdiff(lasso_vars, boruta_vars)
only_boruta <- setdiff(boruta_vars, lasso_vars)

# 保留你原代码的最终推荐规则
if (length(common_vars) >= 3) {
  final_vars  <- common_vars
  final_label <- "LASSO ∩ Boruta 交集"
} else {
  final_vars  <- union_vars
  final_label <- "并集 (交集 < 3, 改用并集)"
}

cat(
  sprintf(
    "LASSO  (%2d): %s\n",
    length(lasso_vars),
    paste(lasso_vars, collapse = ", ")
  )
)
cat(
  sprintf(
    "Boruta (%2d): %s\n",
    length(boruta_vars),
    paste(boruta_vars, collapse = ", ")
  )
)
cat(
  sprintf(
    "交集   (%2d): %s\n",
    length(common_vars),
    ifelse(
      length(common_vars) == 0,
      "无",
      paste(common_vars, collapse = ", ")
    )
  )
)
cat(
  sprintf(
    "并集   (%2d): %s\n",
    length(union_vars),
    paste(union_vars, collapse = ", ")
  )
)
cat(
  sprintf(
    "仅LASSO (%2d): %s\n",
    length(only_lasso),
    ifelse(
      length(only_lasso) == 0,
      "无",
      paste(only_lasso, collapse = ", ")
    )
  )
)
cat(
  sprintf(
    "仅Boruta(%2d): %s\n",
    length(only_boruta),
    ifelse(
      length(only_boruta) == 0,
      "无",
      paste(only_boruta, collapse = ", ")
    )
  )
)

cat(
  sprintf(
    "\n★ 最终推荐 (%s): %d 个\n",
    final_label,
    length(final_vars)
  )
)
cat("  ", paste(final_vars, collapse = ", "), "\n")

# ★ 现行主分析固定9变量（标准缩写）
ORIGINAL_PRIMARY9 <- c(
  "NEU",
  "Intubation",
  "MV",
  "LDH",
  "LYM",
  "BUN",
  "CCI",
  "FIB",
  "Surgery"
)

compare_vars <- unique(c(
  ORIGINAL_PRIMARY9,
  common_vars
))

compare_df <- data.frame(
  Variable = compare_vars,
  In_current_primary9 = compare_vars %in% ORIGINAL_PRIMARY9,
  In_reselected_LASSO_Boruta_intersection =
    compare_vars %in% common_vars,
  stringsAsFactors = FALSE
)

compare_df$Status <- ifelse(
  compare_df$In_current_primary9 &
    compare_df$In_reselected_LASSO_Boruta_intersection,
  "Shared",
  ifelse(
    compare_df$In_current_primary9,
    "Current_primary9_only",
    "Reselected_primary_only"
  )
)

write.csv(
  compare_df,
  "07_主分析重筛_vs_旧9变量_比较.csv",
  row.names = FALSE
)

write.csv(
  data.frame(Variable = common_vars),
  "06_主分析重筛_LASSO_Boruta_intersection.csv",
  row.names = FALSE
)

cat(
  "\n主分析重筛交集与旧9变量完全相同? ",
  ifelse(
    setequal(common_vars, ORIGINAL_PRIMARY9),
    "YES",
    "NO"
  ),
  "\n"
)


# ============================================================
# 8. Age 专项输出
# ============================================================
cat("\n========== 8. Age专项诊断 ==========\n")

age_lasso_rows <- coef_all_df[
  grepl("^Age", coef_all_df$term),
  ,
  drop = FALSE
]

age_boruta_row <- boruta_imp[
  boruta_imp$variable == "Age",
  ,
  drop = FALSE
]

age_diag <- data.frame(
  Metric = c(
    "LASSO lambda.1se coefficient",
    "LASSO selected",
    "Boruta meanImp (Z-score)",
    "Boruta medianImp (Z-score)",
    "Boruta maxImp (Z-score)",
    "Boruta decision"
  ),
  Value = c(
    ifelse(
      nrow(age_lasso_rows) > 0,
      as.character(age_lasso_rows$coefficient[1]),
      NA
    ),
    as.character("Age" %in% lasso_vars),
    ifelse(
      nrow(age_boruta_row) > 0,
      as.character(age_boruta_row$meanImp[1]),
      NA
    ),
    ifelse(
      nrow(age_boruta_row) > 0,
      as.character(age_boruta_row$medianImp[1]),
      NA
    ),
    ifelse(
      nrow(age_boruta_row) > 0,
      as.character(age_boruta_row$maxImp[1]),
      NA
    ),
    ifelse(
      nrow(age_boruta_row) > 0,
      as.character(age_boruta_row$decision[1]),
      NA
    )
  ),
  stringsAsFactors = FALSE
)

print(age_diag)
write.csv(
  age_diag,
  "10_Age_LASSO_Boruta诊断.csv",
  row.names = FALSE
)


# ============================================================
# 9. 保存各阶段数据集
# ============================================================
save_subset <- function(vars, tag) {
  v <- intersect(vars, colnames(dat))
  if (length(v) == 0) return(invisible())

  d <- dat[
    ,
    c(
      "Study_row_id",
      "Primary_split",
      OUTCOME,
      v
    ),
    drop = FALSE
  ]

  fn <- paste0(OUTCOME_CN, "_", tag, ".csv")
  write.csv(d, fn, row.names = FALSE)

  cat(
    "保存:", fn,
    "(", length(v), "个特征)\n"
  )
}

cat("\n保存各阶段数据集：\n")
save_subset(lasso_vars,  "LASSO筛选后数据")
save_subset(boruta_vars, "Boruta筛选后数据")
save_subset(common_vars, "交集数据")
save_subset(union_vars,  "并集数据")
save_subset(final_vars,  "最终推荐数据")


# ============================================================
# 10. 特征筛选总汇总表
# ============================================================
r_map <- setNames(
  scr$df$Spearman_r,
  scr$df$Variable
)
p_map <- setNames(
  scr$df$P_value,
  scr$df$Variable
)
t_map <- setNames(
  scr$df$Type,
  scr$df$Variable
)

summary_vars <- PREDICTORS_AFTER_EXCLUSION

summary_df <- data.frame(
  Variable = summary_vars,
  Group = sapply(summary_vars, get_group),
  Type = ifelse(
    summary_vars %in% names(t_map),
    t_map[summary_vars],
    ""
  ),
  Excluded_zero_variance = ifelse(
    summary_vars %in% zero_vars,
    "√",
    ""
  ),
  Spearman_r = ifelse(
    summary_vars %in% names(r_map),
    r_map[summary_vars],
    NA
  ),
  P_value = ifelse(
    summary_vars %in% names(p_map),
    p_map[summary_vars],
    NA
  ),
  LASSO = ifelse(
    summary_vars %in% lasso_vars,
    "√",
    ""
  ),
  Boruta = ifelse(
    summary_vars %in% boruta_vars,
    "√",
    ""
  ),
  Common = ifelse(
    summary_vars %in% common_vars,
    "√",
    ""
  ),
  Final = ifelse(
    summary_vars %in% final_vars,
    "★",
    ""
  ),
  stringsAsFactors = FALSE
)

abs_r <- abs(summary_df$Spearman_r)
abs_r[is.na(abs_r)] <- -1

summary_df <- summary_df[
  order(
    summary_df$Final == "★",
    summary_df$Common == "√",
    summary_df$LASSO == "√",
    summary_df$Boruta == "√",
    abs_r,
    decreasing = c(
      TRUE, TRUE, TRUE, TRUE, TRUE
    ),
    method = "radix"
  ),
]

print(summary_df, row.names = FALSE)

write.csv(
  summary_df,
  paste0(OUTCOME_CN, "_特征筛选汇总.csv"),
  row.names = FALSE
)


# ============================================================
# 11. Spearman相关性条形图
# ============================================================
cat("\n========== 11. 变量与结局相关性图 ==========\n")

spearman_df <- scr$df[
  ,
  c("Variable", "Spearman_r", "P_value")
]
spearman_df <- spearman_df[
  order(
    abs(spearman_df$Spearman_r),
    decreasing = TRUE
  ),
]

GROUP_COLORS <- c(
  Demographics   = "#4C72B0",
  Comorbidities  = "#C44E52",
  `Life Support` = "#55A868",
  Medications    = "#937860",
  `Lab Tests`    = "#8172B2"
)

png(
  paste0(OUTCOME_CN, "_与结局相关性条形图.png"),
  width = 2400,
  height = max(
    2400,
    length(PREDICTORS) * 90
  ),
  res = 300
)

par(mar = c(5, 9, 3, 2))

ord <- order(abs(spearman_df$Spearman_r))
plot_df <- spearman_df[ord, ]

bar_cols <- GROUP_COLORS[
  sapply(
    plot_df$Variable,
    get_group
  )
]

bp <- barplot(
  plot_df$Spearman_r,
  names.arg = plot_df$Variable,
  horiz = TRUE,
  las = 1,
  col = bar_cols,
  border = "white",
  xlim = c(-0.75, 0.75),
  xlab = "Spearman r (correlation with outcome)",
  main = "",
  cex.names = 0.7,
  cex.axis = 0.9
)

abline(v = 0, col = "black", lwd = 1)
abline(
  v = c(
    -COR_OUTCOME_TH,
    COR_OUTCOME_TH
  ),
  col = "red",
  lty = 2
)

star <- ifelse(
  plot_df$P_value < 0.001,
  "***",
  ifelse(
    plot_df$P_value < 0.01,
    "**",
    ifelse(
      plot_df$P_value < 0.05,
      "*",
      ""
    )
  )
)

text(
  plot_df$Spearman_r +
    ifelse(
      plot_df$Spearman_r >= 0,
      0.04,
      -0.04
    ),
  bp,
  star,
  cex = 0.7,
  adj = ifelse(
    plot_df$Spearman_r >= 0,
    0,
    1
  )
)

legend(
  "bottomright",
  legend = names(GROUP_COLORS),
  fill = GROUP_COLORS,
  cex = 0.7,
  bty = "n"
)

dev.off()


# ============================================================
# 12. 全候选 / 最终变量 Spearman热图
# ============================================================
plot_corr_heatmap <- function(
  vars,
  fname,
  annot_mode = "threshold",
  annot_thresh = 0.50
) {
  if (length(vars) < 2) return(invisible())

  num_df <- data.frame(
    lapply(
      dat[, vars, drop = FALSE],
      function(x) {
        suppressWarnings(
          as.numeric(as.character(x))
        )
      }
    ),
    check.names = FALSE
  )

  colnames(num_df) <- vars

  cm <- suppressWarnings(
    cor(
      num_df,
      use = "pairwise.complete.obs",
      method = "spearman"
    )
  )

  n <- length(vars)

  png(
    fname,
    width = max(2400, n * 110),
    height = max(2400, n * 110),
    res = 300
  )

  par(mar = c(12, 12, 3, 4))

  cols <- colorRampPalette(
    c(
      "#053061", "#2166AC", "#4393C3",
      "#92C5DE", "#F7F7F7", "#F4A582",
      "#D6604D", "#B2182B", "#67001F"
    )
  )(100)

  image(
    1:n,
    1:n,
    t(cm[n:1, ]),
    col = cols,
    zlim = c(-1, 1),
    axes = FALSE,
    xlab = "",
    ylab = "",
    main = ""
  )

  axis(
    1,
    at = 1:n,
    labels = vars,
    las = 2,
    cex.axis = 1.2
  )

  axis(
    2,
    at = 1:n,
    labels = rev(vars),
    las = 1,
    cex.axis = 1.2
  )

  cex_txt <- if (
    annot_mode == "all"
  ) {
    max(
      0.45,
      min(0.85, 10 / n)
    )
  } else {
    0.5
  }

  for (i in 1:n) {
    for (j in 1:n) {
      val <- cm[i, j]
      if (is.na(val)) next

      show <- FALSE
      if (annot_mode == "all") {
        show <- TRUE
      } else if (
        abs(val) >= annot_thresh &&
        i != j
      ) {
        show <- TRUE
      }

      if (show) {
        text(
          j,
          n - i + 1,
          sprintf("%.2f", val),
          cex = cex_txt,
          col = ifelse(
            abs(val) > 0.75,
            "white",
            "black"
          )
        )
      }
    }
  }

  dev.off()
  cat("已保存:", fname, "\n")
}

cat("\n========== 12. 相关性热图 ==========\n")

plot_corr_heatmap(
  PREDICTORS,
  paste0(
    OUTCOME_CN,
    "_相关性热图_主分析全部.png"
  ),
  annot_mode = "threshold",
  annot_thresh = 0.50
)

if (length(final_vars) >= 2) {
  plot_corr_heatmap(
    final_vars,
    paste0(
      OUTCOME_CN,
      "_相关性热图_最终.png"
    ),
    annot_mode = "all"
  )
}


# ============================================================
# 13. 20次重复稳定性分析
#     ★ 同一个固定训练集，不重新抽患者
# ============================================================
if (RUN_STABILITY) {

  cat(
    "\n========== 13. 20次重复稳定性分析 ==========\n"
  )
  cat(
    "说明：每次都使用同一个固定训练集；",
    "不bootstrap、不subsample、不重新划Train/Test。\n"
  )

  # ----------------------------
  # 13A. LASSO 20次
  # ----------------------------
  lasso_repeat <- matrix(
    0L,
    nrow = length(PREDICTORS),
    ncol = N_REPEATS,
    dimnames = list(
      PREDICTORS,
      paste0("Run", 1:N_REPEATS)
    )
  )

  lasso_run_detail <- data.frame(
    Run = integer(),
    Seed = integer(),
    lambda_min = numeric(),
    lambda_1se = numeric(),
    N_selected = integer(),
    Selected = character(),
    stringsAsFactors = FALSE
  )

  for (r in 1:N_REPEATS) {
    seed_r <- LASSO_SEED + r - 1
    cat(
      "LASSO repeat",
      r, "/", N_REPEATS,
      "| seed =", seed_r, "\n"
    )

    set.seed(seed_r)
    cv_r <- cv.glmnet(
      lasso_x,
      lasso_y,
      family = "binomial",
      alpha = 1,
      nfolds = 10,
      maxit = 1e5
    )

    c_r <- coef(
      cv_r,
      s = "lambda.1se"
    )

    c_df <- data.frame(
      term = rownames(c_r),
      coefficient = as.numeric(c_r),
      stringsAsFactors = FALSE
    )

    c_df <- subset(
      c_df,
      coefficient != 0 &
      term != "(Intercept)"
    )

    vars_r <- unique(
      vapply(
        c_df$term,
        map_dummy_to_var,
        character(1),
        vars = PREDICTORS
      )
    )

    lasso_repeat[
      rownames(lasso_repeat) %in% vars_r,
      r
    ] <- 1L

    lasso_run_detail <- rbind(
      lasso_run_detail,
      data.frame(
        Run = r,
        Seed = seed_r,
        lambda_min = cv_r$lambda.min,
        lambda_1se = cv_r$lambda.1se,
        N_selected = length(vars_r),
        Selected = paste(
          vars_r,
          collapse = ";"
        ),
        stringsAsFactors = FALSE
      )
    )
  }

  lasso_stability <- data.frame(
    Variable = rownames(lasso_repeat),
    Selected_Count = rowSums(lasso_repeat),
    Not_Selected_Count =
      N_REPEATS - rowSums(lasso_repeat),
    Selection_Frequency =
      rowMeans(lasso_repeat),
    stringsAsFactors = FALSE
  )

  lasso_stability$Stability_Class <- ifelse(
    lasso_stability$Selection_Frequency >= 0.80,
    "Stably selected (>=80%)",
    ifelse(
      lasso_stability$Selection_Frequency < 0.20,
      "Stably rejected (<20%)",
      "Variable (20%-80%)"
    )
  )

  write.csv(
    lasso_stability,
    "08_LASSO_主分析重筛_20次重复稳定性.csv",
    row.names = FALSE
  )

  write.csv(
    lasso_run_detail,
    "08b_LASSO_主分析重筛_20次运行明细.csv",
    row.names = FALSE
  )

  # ----------------------------
  # 13B. Boruta 20次
  # ----------------------------
  boruta_repeat <- matrix(
    NA_character_,
    nrow = length(PREDICTORS),
    ncol = N_REPEATS,
    dimnames = list(
      PREDICTORS,
      paste0("Run", 1:N_REPEATS)
    )
  )

  for (r in 1:N_REPEATS) {
    seed_r <- BORUTA_SEED + r - 1

    cat(
      "Boruta repeat",
      r, "/", N_REPEATS,
      "| seed =", seed_r, "\n"
    )

    set.seed(seed_r)

    b_r <- Boruta(
      as.formula(
        paste0("`", OUTCOME, "` ~ .")
      ),
      data = train_complete,
      doTrace = 0,
      maxRuns = 100,
      num.trees = 1000
    )

    b_r_fixed <- TentativeRoughFix(b_r)

    # Boruta包没有 finalDecision() 这个函数。
    # TentativeRoughFix 后通过 attStats() 读取每个变量的最终 decision。
    stats_r <- attStats(b_r_fixed)
    dec_r <- as.character(stats_r$decision)
    names(dec_r) <- rownames(stats_r)

    for (v in PREDICTORS) {
      if (v %in% names(dec_r)) {
        boruta_repeat[v, r] <- dec_r[v]
      }
    }
  }

  boruta_stability <- data.frame(
    Variable = rownames(boruta_repeat),
    Confirmed_Count = rowSums(
      boruta_repeat == "Confirmed",
      na.rm = TRUE
    ),
    Rejected_Count = rowSums(
      boruta_repeat == "Rejected",
      na.rm = TRUE
    ),
    Tentative_Count = rowSums(
      boruta_repeat == "Tentative",
      na.rm = TRUE
    ),
    stringsAsFactors = FALSE
  )

  boruta_stability$Selection_Frequency <-
    boruta_stability$Confirmed_Count /
    N_REPEATS

  boruta_stability$Stability_Class <- ifelse(
    boruta_stability$Selection_Frequency >= 0.80,
    "Stably selected (>=80%)",
    ifelse(
      boruta_stability$Selection_Frequency < 0.20,
      "Stably rejected (<20%)",
      "Variable (20%-80%)"
    )
  )

  write.csv(
    boruta_stability,
    "09_Boruta_主分析重筛_20次重复稳定性.csv",
    row.names = FALSE
  )

  # ----------------------------
  # 13C. Age重复稳定性补充
  # ----------------------------
  age_lasso_freq <- lasso_stability$Selection_Frequency[
    lasso_stability$Variable == "Age"
  ]

  age_boruta_freq <- boruta_stability$Selection_Frequency[
    boruta_stability$Variable == "Age"
  ]

  age_repeat <- data.frame(
    Metric = c(
      "LASSO 20-run selection frequency",
      "Boruta 20-run confirmation frequency"
    ),
    Value = c(
      ifelse(
        length(age_lasso_freq) == 1,
        age_lasso_freq,
        NA
      ),
      ifelse(
        length(age_boruta_freq) == 1,
        age_boruta_freq,
        NA
      )
    )
  )

  write.csv(
    age_repeat,
    "10b_Age_20次重复稳定性.csv",
    row.names = FALSE
  )
}


# ============================================================
# 14. 方法说明 / 总览
# ============================================================
method_note <- c(
  "Primary feature-selection reanalysis after prespecified exclusions",
  "",
  paste0(
    "Input file: ",
    INPUT_FILE
  ),
  "",
  "Key design:",
  "- The original primary-analysis Train/Test identity was preserved.",
  "- Train: n=2357, events=950.",
  "- Test: n=1011, events=407.",
  "- No new random train/test split was created for this primary feature-selection reanalysis.",
  "- Broad-spectrum antibiotics (BSA) were excluded a priori before feature selection because of potentially endogenous timing.",
  "- Immunosuppressant use was excluded a priori because it had zero variance in the entire cohort (0/3368).
- The standalone Heart disease field was excluded before feature selection on data-quality grounds because it was known not to validly represent overall cardiac comorbidity; it was distinct from ICD-10-derived Charlson cardiac components.",
  "- Spearman correlations were descriptive only and were not used to screen variables.",
  paste0("- Candidate count after the two a priori exclusions plus the Heart disease data-quality exclusion: ", length(PREDICTORS_AFTER_EXCLUSION), " (expected 35)."),
  "- Training-set zero-variance variables were then removed as a technical preprocessing step before LASSO/Boruta.",
  paste0(
    "- Training-set zero-variance variables removed: ",
    ifelse(
      length(zero_vars) == 0,
      "none",
      paste(zero_vars, collapse = ", ")
    )
  ),
  paste0(
    "- Non-zero-variance candidates actually entered into LASSO/Boruta: ",
    length(PREDICTORS)
  ),
  "",
  "LASSO:",
  "- Binomial LASSO, alpha=1.",
  "- 10-fold cross-validation.",
  "- lambda.1se used for feature selection.",
  "",
  "Boruta:",
  "- Random-forest Boruta.",
  "- maxRuns=100.",
  "- num.trees=1000.",
  "- TentativeRoughFix was applied.",
  "- Only confirmed variables after rough fix were counted as selected.",
  "",
  "Repeated stability analysis:",
  paste0(
    "- Number of repeated runs: ",
    ifelse(
      RUN_STABILITY,
      N_REPEATS,
      0
    )
  ),
  "- The same fixed training cohort was used in every repetition.",
  "- No patient bootstrap/subsampling and no train/test repartitioning were performed.",
  "- LASSO repeats varied the random CV fold allocation.",
  "- Boruta repeats varied only the stochastic random seed.",
  "",
  paste0(
    "Primary reselected LASSO-Boruta intersection: ",
    paste(
      common_vars,
      collapse = ", "
    )
  ),
  paste0(
    "Identical to legacy constrained 9 predictors: ",
    ifelse(
      setequal(
        common_vars,
        ORIGINAL_PRIMARY9
      ),
      "YES",
      "NO"
    )
  )
)

writeLines(
  method_note,
  "README_主分析HeartDisease排除_34候选变量重筛.txt"
)


# ============================================================
# 15. 总览
# ============================================================
sep <- paste0(
  rep("=", 60),
  collapse = ""
)

cat("\n\n", sep, "\n", sep = "")
cat("      主分析HeartDisease数据质量排除后34候选变量重筛完成总览\n")
cat(sep, "\n")

cat(
  "原候选特征数:",
  length(ALL_PREDICTORS), "\n"
)
cat(
  "先验排除BSA+Immunosuppressants后:",
  length(PREDICTORS_AFTER_EXCLUSION), "\n"
)
cat(
  "去零方差后实际筛选:",
  length(PREDICTORS), "\n"
)
cat(
  "LASSO选中:",
  length(lasso_vars), "\n"
)
cat(
  "Boruta选中:",
  length(boruta_vars), "\n"
)
cat(
  "交集:",
  length(common_vars), "\n"
)
cat(
  "最终推荐:",
  length(final_vars), "\n"
)

cat("\n最终变量列表:\n")
for (v in final_vars) {
  cat(
    "  ·", v,
    "[", get_group(v), "]\n"
  )
}

cat(
  "\n主分析重筛交集与旧9变量完全相同? ",
  ifelse(
    setequal(
      common_vars,
      ORIGINAL_PRIMARY9
    ),
    "YES",
    "NO"
  ),
  "\n"
)

cat("\n所有文件已保存至:\n  ", SAVE_PATH, "\n")
cat(sep, "\n")
cat("分析完成!\n")
cat(sep, "\n")

cat("\n请优先把以下结果发给我：\n")
cat("  00_主分析先验排除变量与原因.csv\n")
cat("  01_主分析35候选变量_零方差与稀有度核查.csv\n")
cat("  肺部感染_主分析重筛_LASSO_lambda1se全部系数.csv\n")
cat("  肺部感染_主分析重筛_Boruta重要性.csv\n")
cat("  06_主分析重筛_LASSO_Boruta_intersection.csv\n")
cat("  07_主分析重筛_vs_旧9变量_比较.csv\n")
cat("  10_Age_LASSO_Boruta诊断.csv\n")
if (RUN_STABILITY) {
  cat("  08_LASSO_主分析重筛_20次重复稳定性.csv\n")
  cat("  09_Boruta_主分析重筛_20次重复稳定性.csv\n")
  cat("  10b_Age_20次重复稳定性.csv\n")
}
