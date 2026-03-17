################################################################################
# Unified CH4 temperature-response workflow for paddy and wetland ecosystems
# Updated for robust end-to-end analysis (meta + ML + spatial projection)
################################################################################

required_packages <- c(
  "readxl", "dplyr", "tidyr", "stringr", "purrr", "ggplot2", "forcats",
  "metafor", "lme4", "mgcv", "caret", "ranger", "xgboost", "nnet",
  "yardstick", "terra", "geodata", "tibble"
)

optional_packages <- c("brms", "vip")

install_and_load_packages <- function(pkgs = required_packages) {
  missing <- pkgs[!pkgs %in% rownames(installed.packages())]
  if (length(missing) > 0) {
    install.packages(missing, repos = "https://cloud.r-project.org")
  }
  invisible(lapply(pkgs, function(p) suppressPackageStartupMessages(library(p, character.only = TRUE))))
}

load_optional_packages <- function(pkgs = optional_packages) {
  loaded <- purrr::map_lgl(pkgs, function(p) {
    suppressWarnings(require(p, character.only = TRUE, quietly = TRUE))
  })
  stats::setNames(loaded, pkgs)
}

safe_factor <- function(x) {
  x <- ifelse(is.na(x) | x == "", "Unknown", as.character(x))
  as.factor(x)
}

find_col <- function(df, candidates) {
  nm <- names(df)
  hit <- candidates[candidates %in% nm]
  if (length(hit) == 0) return(NA_character_)
  hit[1]
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

  if (!all(c("longitude", "latitude") %in% names(df))) {
    stop("Input data must contain longitude and latitude columns.")
  }

  pts <- terra::vect(df, geom = c("longitude", "latitude"), crs = "EPSG:4326")
  vars <- c("clay", "soc", "nitrogen", "phh2o")

  soil_layers <- purrr::map(vars, function(v) geodata::soil_world(var = v, path = cache_dir))
  names(soil_layers) <- vars

  extracted <- purrr::imap_dfc(soil_layers, function(r, nm) {
    val <- terra::extract(r, pts)[, 2, drop = TRUE]
    tibble::tibble(!!nm := as.numeric(val))
  })

  out <- bind_cols(df, extracted) %>%
    mutate(
      clay = clay / 10,
      soc = soc / 10,
      tn = nitrogen / 100,
      ph = phh2o / 10,
      som = soc * 1.724
    ) %>%
    select(-nitrogen, -phh2o)

  classify_soil(out)
}

create_effect_size <- function(df) {
  req <- c("Me", "Ma", "SDe", "Ne", "SDa", "Na")
  miss <- req[!req %in% names(df)]
  if (length(miss) > 0) stop(paste("Missing effect-size fields:", paste(miss, collapse = ", ")))

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
    sm <- coef(summary(fit))
    tibble::tibble(
      moderator = m,
      term = rownames(sm),
      estimate = sm[, "estimate"],
      se = sm[, "se"],
      pval = sm[, "pval"]
    )
  })

  write.csv(mods_out, file.path(out_dir, "meta_moderators.csv"), row.names = FALSE)
}

detect_threshold <- function(df, out_file) {
  x <- df$increase
  y <- df$yi
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]; y <- y[ok]
  if (length(x) < 30) return(NULL)

  grid <- seq(stats::quantile(x, 0.2), stats::quantile(x, 0.8), length.out = 40)
  best <- NULL
  best_rss <- Inf

  for (bp in grid) {
    left <- pmax(0, bp - x)
    right <- pmax(0, x - bp)
    fit <- lm(y ~ left + right)
    rss <- sum(residuals(fit)^2)
    if (rss < best_rss) {
      best_rss <- rss
      best <- list(bp = bp, fit = fit, rss = rss)
    }
  }

  if (!is.null(best)) {
    res <- data.frame(
      breakpoint = best$bp,
      rss = best$rss,
      left_slope = coef(best$fit)["left"],
      right_slope = coef(best$fit)["right"]
    )
    write.csv(res, out_file, row.names = FALSE)
  }
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

run_predictive_models <- function(df, ecosystem, out_dir, optional_state) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  set.seed(123)

  warming_col <- find_col(df, c("increase", "Increase", "warming"))
  facility_col <- find_col(df, c("facility", "Facility"))
  warming_type_col <- find_col(df, c("Warming type", "warming_type", "warmingtype"))

  predictors <- c(warming_col, "latitude", "longitude", "MAT", "MAP", "duration", "clay", "soc", "tn", "ph", facility_col, warming_type_col)
  if (ecosystem == "paddy") predictors <- unique(c(predictors, "napplication", "watermanagement", "strawincorporation"))
  if (ecosystem == "wetland") predictors <- unique(c(predictors, "wetlandsubclass", "watertable", "plantspecies"))
  predictors <- predictors[!is.na(predictors) & predictors %in% names(df)]

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
  lmm_fit <- if (!is.na(facility_col) && facility_col %in% names(train)) {
    lme4::lmer(stats::as.formula(paste("yi ~ . + (1|", facility_col, ")")), data = train)
  } else NULL
  gam_fit <- mgcv::gam(fml, data = train, method = "REML")

  rf_fit <- caret::train(fml, data = train, method = "ranger", trControl = ctrl, tuneLength = 8, importance = "impurity")
  xgb_fit <- caret::train(fml, data = train, method = "xgbTree", trControl = ctrl, tuneLength = 6)
  nn_fit <- caret::train(fml, data = train, method = "nnet", trControl = ctrl, tuneLength = 6, trace = FALSE)

  models <- list(lm = lm_fit, gam = gam_fit, rf = rf_fit, xgb = xgb_fit, nnet = nn_fit)
  if (!is.null(lmm_fit)) models$lmm <- lmm_fit

  if (isTRUE(optional_state[["brms"]]) && !is.na(facility_col) && facility_col %in% names(train) && !is.na(warming_col)) {
    bayes_fit <- tryCatch(
      brms::brm(
        formula = stats::as.formula(paste("yi ~ s(", warming_col, ") + s(MAT) + s(MAP) + (1|", facility_col, ")")),
        data = train, family = gaussian(), chains = 2, iter = 2000, cores = 2, refresh = 0
      ),
      error = function(e) NULL
    )
    if (!is.null(bayes_fit)) models$bayes <- bayes_fit
  }

  metrics <- purrr::imap_dfr(models, function(m, n) {
    pred_train <- if (n == "bayes") colMeans(predict(m, newdata = train)) else predict(m, newdata = train)
    pred_test <- if (n == "bayes") colMeans(predict(m, newdata = test)) else predict(m, newdata = test)
    bind_rows(model_metrics(train$yi, pred_train, "train", n), model_metrics(test$yi, pred_test, "test", n))
  })
  write.csv(metrics, file.path(out_dir, "model_metrics.csv"), row.names = FALSE)

  best_model_name <- metrics %>% filter(dataset == "test") %>% arrange(rmse) %>% slice(1) %>% pull(model)
  saveRDS(models[[best_model_name]], file.path(out_dir, "best_model.rds"))
  writeLines(best_model_name, file.path(out_dir, "best_model_name.txt"))

  if (isTRUE(optional_state[["vip"]]) && best_model_name %in% c("rf", "xgb")) {
    imp <- vip::vi(models[[best_model_name]])
    write.csv(imp, file.path(out_dir, "best_model_importance.csv"), row.names = FALSE)
  }

  if (!is.na(warming_col) && warming_col %in% names(test)) {
    grid_df <- test
    grid_df[[warming_col]] <- seq(min(df[[warming_col]], na.rm = TRUE), max(df[[warming_col]], na.rm = TRUE), length.out = 200)
    for (i in setdiff(names(grid_df), c("yi", warming_col))) {
      if (is.numeric(grid_df[[i]])) grid_df[[i]] <- median(df[[i]], na.rm = TRUE)
      if (is.factor(grid_df[[i]])) grid_df[[i]] <- levels(grid_df[[i]])[1]
    }
    p <- if (best_model_name == "bayes") colMeans(predict(models[[best_model_name]], newdata = grid_df)) else predict(models[[best_model_name]], newdata = grid_df)
    response_curve <- tibble::tibble(increase = grid_df[[warming_col]], pred_lnrr = p)
    write.csv(response_curve, file.path(out_dir, "warming_response_curve.csv"), row.names = FALSE)

    plt <- ggplot(response_curve, aes(increase, pred_lnrr)) +
      geom_line(linewidth = 1.1, color = "#D7191C") +
      theme_bw() +
      labs(x = "Warming increment (°C)", y = "Predicted LnRR", title = "Estimated non-linear temperature response")
    ggsave(file.path(out_dir, "warming_response_curve.png"), plt, width = 6, height = 4, dpi = 300)

    detect_threshold(df, file.path(out_dir, "warming_breakpoint_estimate.csv"))
  }

  list(best_model = models[[best_model_name]], best_model_name = best_model_name)
}

run_spatial_prediction <- function(best_model, ecosystem, out_dir) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  china_ext <- terra::ext(73, 136, 18, 54)
  wc <- geodata::worldclim_global(var = c("tavg", "prec"), res = 10, path = file.path(out_dir, "cache"))
  wc <- terra::crop(wc, china_ext)

  tavg <- terra::app(wc[[grep("tavg", names(wc))]], fun = mean, na.rm = TRUE); names(tavg) <- "MAT"
  prec <- terra::app(wc[[grep("prec", names(wc))]], fun = sum, na.rm = TRUE); names(prec) <- "MAP"

  add_const <- function(r, nm, val) {x <- r[[1]] * 0 + val; names(x) <- nm; x}
  common <- rast(c(
    tavg,
    prec,
    add_const(tavg, "increase", 0),
    add_const(tavg, "latitude", 35),
    add_const(tavg, "longitude", 104),
    add_const(tavg, "duration", 365),
    add_const(tavg, "clay", 30),
    add_const(tavg, "soc", 20),
    add_const(tavg, "tn", 1.5),
    add_const(tavg, "ph", 6.8)
  ))

  future <- common
  future[["increase"]] <- future[["increase"]] + 3

  curr_pred <- terra::predict(common, best_model)
  fut_pred <- terra::predict(future, best_model)
  delta <- fut_pred - curr_pred

  terra::writeRaster(curr_pred, file.path(out_dir, paste0(ecosystem, "_lnrr_current.tif")), overwrite = TRUE)
  terra::writeRaster(fut_pred, file.path(out_dir, paste0(ecosystem, "_lnrr_future_plus3C.tif")), overwrite = TRUE)
  terra::writeRaster(delta, file.path(out_dir, paste0(ecosystem, "_lnrr_delta_plus3C.tif")), overwrite = TRUE)
}

run_full_workflow <- function(input_file, ecosystem, output_root = "outputs") {
  install_and_load_packages()
  optional_state <- load_optional_packages()

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
  model_res <- run_predictive_models(es, ecosystem, file.path(out_dir, "03_models"), optional_state)
  run_spatial_prediction(model_res$best_model, ecosystem, file.path(out_dir, "04_spatial"))

  message("Workflow completed for: ", ecosystem)
}
