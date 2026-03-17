################################################################################
# Unified CH4 temperature-response workflow for paddy and wetland ecosystems
# Author: Codex agent
################################################################################

required_packages <- c(
  "readxl", "dplyr", "tidyr", "stringr", "purrr", "ggplot2", "forcats",
  "metafor", "lme4", "mgcv", "brms", "caret", "randomForest", "xgboost",
  "ranger", "nnet", "pdp", "vip", "yardstick", "terra", "geodata", "sf",
  "tibble", "writexl"
)

install_and_load_packages <- function(pkgs = required_packages) {
  missing <- pkgs[!pkgs %in% rownames(installed.packages())]
  if (length(missing) > 0) {
    install.packages(missing, repos = "https://cloud.r-project.org")
  }
  invisible(lapply(pkgs, function(p) suppressPackageStartupMessages(library(p, character.only = TRUE))))
}

safe_factor <- function(x) {
  x <- ifelse(is.na(x) | x == "", "Unknown", as.character(x))
  as.factor(x)
}

classify_soil <- function(df) {
  df %>%
    mutate(
      clay_class = cut(clay, breaks = c(-Inf, 15, 35, 60, Inf),
                       labels = c("Sandy/Loamy", "Clay loam", "Clay", "Heavy clay")),
      soc_class = cut(soc, breaks = c(-Inf, 10, 20, 40, Inf),
                      labels = c("Low", "Moderate", "High", "Very high")),
      tn_class = cut(tn, breaks = c(-Inf, 0.5, 1.5, 3, Inf),
                     labels = c("Low", "Moderate", "High", "Very high")),
      ph_class = cut(ph, breaks = c(-Inf, 5.5, 7.5, 8.5, Inf),
                     labels = c("Acidic", "Neutral", "Alkaline", "Strong alkaline"))
    )
}

fetch_soilgrids_at_points <- function(df, cache_dir = "outputs/cache") {
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

  pts <- terra::vect(df, geom = c("longitude", "latitude"), crs = "EPSG:4326")
  vars <- c("clay", "soc", "nitrogen", "phh2o")

  soil_layers <- purrr::map(vars, function(v) {
    geodata::soil_world(var = v, path = cache_dir)
  })

  names(soil_layers) <- vars

  extracted <- purrr::imap_dfc(soil_layers, function(r, nm) {
    val <- terra::extract(r, pts)[, 2, drop = TRUE]
    tibble::tibble(!!nm := as.numeric(val))
  })

  # Unit harmonization (SoilGrids default units)
  out <- bind_cols(df, extracted) %>%
    mutate(
      clay = clay / 10,           # g/kg to %
      soc = soc / 10,             # dg/kg to g/kg
      tn = nitrogen / 100,        # cg/kg to g/kg
      ph = phh2o / 10,            # pH*10 to pH
      som = soc * 1.724           # SOC to SOM conversion (Van Bemmelen factor)
    ) %>%
    select(-nitrogen, -phh2o)

  classify_soil(out)
}

create_effect_size <- function(df) {
  df %>%
    mutate(
      yi = log(Me / Ma),
      vi = (SDe^2) / (Ne * Me^2) + (SDa^2) / (Na * Ma^2)
    ) %>%
    filter(is.finite(yi), is.finite(vi), vi > 0)
}

run_data_quality <- function(df, out_dir) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  missing_tbl <- tibble::tibble(
    variable = names(df),
    missing_n = colSums(is.na(df)),
    missing_pct = round(100 * missing_n / nrow(df), 2)
  )
  write.csv(missing_tbl, file.path(out_dir, "missing_summary.csv"), row.names = FALSE)

  p1 <- ggplot(df, aes(x = yi)) +
    geom_histogram(bins = 30, fill = "#2C7BB6", color = "white") +
    theme_bw() +
    labs(x = "LnRR", y = "Count", title = "Distribution of CH4 response (LnRR)")
  ggsave(file.path(out_dir, "lnrr_histogram.png"), p1, width = 6, height = 4, dpi = 300)

  shapiro_n <- min(5000, nrow(df))
  shapiro_result <- shapiro.test(sample(df$yi, shapiro_n))
  capture.output(shapiro_result, file = file.path(out_dir, "lnrr_normality_test.txt"))
}

run_meta_analysis <- function(df, ecosystem, out_dir) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  df <- df %>% mutate(study = safe_factor(`NO.`))

  overall <- metafor::rma(yi = yi, vi = vi, data = df, method = "REML")
  capture.output(summary(overall), file = file.path(out_dir, "meta_overall.txt"))

  moderators <- if (ecosystem == "paddy") {
    c("MAT1", "MAP1", "napplication1", "watermanagement", "facility", "increase1", "ph_class", "soc_class")
  } else {
    c("MAT1", "MAP1", "watertable1", "facility", "increase1", "wetlandsubclass", "ph_class", "soc_class")
  }

  mods_out <- purrr::map_dfr(moderators, function(m) {
    if (!m %in% names(df)) return(NULL)
    d <- df %>% mutate(tmp = safe_factor(.data[[m]]))
    fit <- metafor::rma(yi, vi, mods = ~ tmp - 1, data = d, method = "REML")
    tibble::tibble(
      moderator = m,
      term = rownames(coef(summary(fit))),
      estimate = coef(summary(fit))[, "estimate"],
      se = coef(summary(fit))[, "se"],
      pval = coef(summary(fit))[, "pval"]
    )
  })

  write.csv(mods_out, file.path(out_dir, "meta_moderators.csv"), row.names = FALSE)
}

model_metrics <- function(obs, pred, dataset_name, model_name) {
  tibble::tibble(
    dataset = dataset_name,
    model = model_name,
    rmse = yardstick::rmse_vec(obs, pred),
    rsq = yardstick::rsq_vec(obs, pred),
    mae = yardstick::mae_vec(obs, pred)
  )
}

run_predictive_models <- function(df, ecosystem, out_dir) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  set.seed(123)
  predictors <- c(
    "increase", "latitude", "longitude", "MAT", "MAP", "duration",
    "clay", "soc", "tn", "ph", "facility", "Warming type"
  )
  predictors <- predictors[predictors %in% names(df)]

  if (ecosystem == "paddy") {
    predictors <- unique(c(predictors, "napplication", "watermanagement", "strawincorporation"))
  } else {
    predictors <- unique(c(predictors, "wetlandsubclass", "watertable", "plantspecies"))
  }
  predictors <- predictors[predictors %in% names(df)]

  model_df <- df %>%
    select(all_of(c("yi", predictors))) %>%
    mutate(across(where(is.character), safe_factor)) %>%
    mutate(across(where(is.factor), forcats::fct_explicit_na, na_level = "Unknown")) %>%
    tidyr::drop_na(yi)

  idx <- caret::createDataPartition(model_df$yi, p = 0.8, list = FALSE)
  train <- model_df[idx, ]
  test <- model_df[-idx, ]

  ctrl <- caret::trainControl(method = "repeatedcv", number = 5, repeats = 2)

  fml <- as.formula("yi ~ .")

  lm_fit <- lm(fml, data = train)
  lmm_fit <- lme4::lmer(update(fml, . ~ . + (1 | facility)), data = train)
  gam_fit <- mgcv::gam(fml, data = train, method = "REML")

  rf_fit <- caret::train(fml, data = train, method = "ranger", trControl = ctrl,
                         tuneLength = 10, importance = "impurity")
  xgb_fit <- caret::train(fml, data = train, method = "xgbTree", trControl = ctrl,
                          tuneLength = 8)
  nn_fit <- caret::train(fml, data = train, method = "nnet", trControl = ctrl,
                         tuneLength = 8, trace = FALSE)

  bayes_fit <- tryCatch(
    brms::brm(
      formula = bf(yi ~ s(increase) + s(MAT) + s(MAP) + (1 | facility)),
      data = train,
      family = gaussian(),
      chains = 2, iter = 2000, cores = 2, refresh = 0
    ),
    error = function(e) NULL
  )

  models <- list(
    lm = lm_fit,
    lmm = lmm_fit,
    gam = gam_fit,
    rf = rf_fit,
    xgb = xgb_fit,
    nnet = nn_fit
  )
  if (!is.null(bayes_fit)) models$bayes <- bayes_fit

  metrics <- purrr::imap_dfr(models, function(m, n) {
    pred_train <- if (n == "bayes") colMeans(predict(m, newdata = train)) else predict(m, newdata = train)
    pred_test <- if (n == "bayes") colMeans(predict(m, newdata = test)) else predict(m, newdata = test)
    bind_rows(
      model_metrics(train$yi, pred_train, "train", n),
      model_metrics(test$yi, pred_test, "test", n)
    )
  })

  write.csv(metrics, file.path(out_dir, "model_metrics.csv"), row.names = FALSE)

  best_model_name <- metrics %>%
    filter(dataset == "test") %>%
    arrange(rmse) %>%
    slice(1) %>%
    pull(model)

  saveRDS(models[[best_model_name]], file.path(out_dir, "best_model.rds"))
  writeLines(best_model_name, file.path(out_dir, "best_model_name.txt"))

  # Importance for tree models
  if (best_model_name %in% c("rf", "xgb")) {
    imp <- vip::vi(models[[best_model_name]])
    write.csv(imp, file.path(out_dir, "best_model_importance.csv"), row.names = FALSE)
  }

  # Non-linearity diagnostic: response curve of warming increment
  grid_df <- test
  grid_df$increase <- seq(min(df$increase, na.rm = TRUE), max(df$increase, na.rm = TRUE), length.out = 200)
  for (i in setdiff(names(grid_df), c("yi", "increase"))) {
    if (is.numeric(grid_df[[i]])) grid_df[[i]] <- median(df[[i]], na.rm = TRUE)
    if (is.factor(grid_df[[i]])) grid_df[[i]] <- levels(grid_df[[i]])[1]
  }

  p <- if (best_model_name == "bayes") colMeans(predict(models[[best_model_name]], newdata = grid_df))
  else predict(models[[best_model_name]], newdata = grid_df)

  response_curve <- tibble::tibble(increase = grid_df$increase, pred_lnrr = p)
  write.csv(response_curve, file.path(out_dir, "warming_response_curve.csv"), row.names = FALSE)

  plt <- ggplot(response_curve, aes(increase, pred_lnrr)) +
    geom_line(linewidth = 1.1, color = "#D7191C") +
    theme_bw() +
    labs(x = "Warming increment (°C)", y = "Predicted LnRR", title = "Estimated non-linear temperature response")
  ggsave(file.path(out_dir, "warming_response_curve.png"), plt, width = 6, height = 4, dpi = 300)

  list(best_model = models[[best_model_name]], best_model_name = best_model_name)
}

run_spatial_prediction <- function(best_model, ecosystem, out_dir, climate_source = "worldclim") {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  # China extent
  china_ext <- terra::ext(73, 136, 18, 54)

  wc <- geodata::worldclim_global(var = c("tavg", "prec"), res = 10, path = file.path(out_dir, "cache"))
  wc <- terra::crop(wc, china_ext)

  # derive annual summaries
  tavg <- terra::app(wc[[grep("tavg", names(wc))]], fun = mean, na.rm = TRUE)
  prec <- terra::app(wc[[grep("prec", names(wc))]], fun = sum, na.rm = TRUE)
  names(tavg) <- "MAT"
  names(prec) <- "MAP"

  # baseline warming = 0 and scenario warming = +3C
  increase0 <- tavg * 0; names(increase0) <- "increase"
  increase3 <- tavg * 0 + 3; names(increase3) <- "increase"

  # placeholders for features unavailable as spatial layers
  add_const <- function(r, nm, val) {
    x <- r[[1]] * 0 + val
    names(x) <- nm
    x
  }

  stack_common <- c(tavg, prec, increase0,
                    add_const(tavg, "latitude", terra::yFromRow(tavg, 1)[1]),
                    add_const(tavg, "longitude", terra::xFromCol(tavg, 1)[1]),
                    add_const(tavg, "duration", 365),
                    add_const(tavg, "clay", 30),
                    add_const(tavg, "soc", 20),
                    add_const(tavg, "tn", 1.5),
                    add_const(tavg, "ph", 6.8))

  curr <- rast(stack_common)
  fut <- rast(stack_common); fut[["increase"]] <- increase3

  # spatial predict
  curr_pred <- terra::predict(curr, best_model)
  fut_pred <- terra::predict(fut, best_model)
  delta <- fut_pred - curr_pred

  terra::writeRaster(curr_pred, file.path(out_dir, paste0(ecosystem, "_lnrr_current.tif")), overwrite = TRUE)
  terra::writeRaster(fut_pred, file.path(out_dir, paste0(ecosystem, "_lnrr_future_plus3C.tif")), overwrite = TRUE)
  terra::writeRaster(delta, file.path(out_dir, paste0(ecosystem, "_lnrr_delta_plus3C.tif")), overwrite = TRUE)
}

run_full_workflow <- function(input_file, ecosystem, output_root = "outputs") {
  install_and_load_packages()

  out_dir <- file.path(output_root, ecosystem)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  raw <- readxl::read_excel(input_file)
  raw <- raw %>% mutate(across(where(is.character), stringr::str_trim))

  enriched <- fetch_soilgrids_at_points(raw, cache_dir = file.path(out_dir, "cache"))
  write.csv(enriched, file.path(out_dir, paste0(ecosystem, "_with_soil.csv")), row.names = FALSE)

  es <- create_effect_size(enriched)
  write.csv(es, file.path(out_dir, paste0(ecosystem, "_effectsize.csv")), row.names = FALSE)

  run_data_quality(es, file.path(out_dir, "01_data_quality"))
  run_meta_analysis(es, ecosystem, file.path(out_dir, "02_meta"))

  model_res <- run_predictive_models(es, ecosystem, file.path(out_dir, "03_models"))
  run_spatial_prediction(model_res$best_model, ecosystem, file.path(out_dir, "04_spatial"))

  message("Workflow completed for: ", ecosystem)
}
