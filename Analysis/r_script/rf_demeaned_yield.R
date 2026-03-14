library(tidyverse)
library("ranger")

data_yield <- read.csv("C:/Users/lukas/Documents/Studium/Remote_Sensing_Products/project/RSapp_project/Analysis/merged_csv_ger/model_table_ww_2017_2023_ger.csv")


data_yield <- data_yield %>%
  group_by(nuts_id) %>%
  mutate(
    yield_value_demeaned = yield_value - mean(yield_value)
  ) %>%
  ungroup()


colnames(data_yield)

target <- "yield_value_demeaned"

predictors <- setdiff(
  colnames(data_yield),
  c("nuts_id", "year", "yield_value", "yield_prev", target)
)

rf_model <- ranger(
  formula = as.formula(
    paste(target, "~", paste(predictors, collapse = " + "))
  ),
  data = data_yield,
  num.trees = 1000,
  mtry = floor(sqrt(length(predictors))),
  importance = "permutation",
  seed = 1,
  oob.error = TRUE
)


y <- data_yield[[target]]
y_pred <- rf_model$predictions
RSS <- sum((y - y_pred)^2)
TSS <- sum((y-mean(y))^2)
oob_r2 <- 1-(RSS/TSS)


cat("OOB R²:", oob_r2, "\n")

years <- sort(unique(data_yield$year))

all_predictions <- NULL

for (yr in years) {
  
  train_data <- data_yield %>% filter(year != yr)
  test_data  <- data_yield %>% filter(year == yr)
  
  rf_cv <- ranger(
    formula = as.formula(
      paste(target, "~", paste(predictors, collapse = " + "))
    ),
    data = train_data,
    num.trees = 1000,
    mtry = floor(sqrt(length(predictors))),
    seed = 42
  )
  preds <- predict(rf_cv, data = test_data)$predictions
  
  df_pred <- data.frame(nuts_id = test_data$nuts_id, year = yr, y = test_data[[target]], y_pred =  preds)

  y_cv <- test_data[[target]]
  y_pred_cv <- preds
  RSS_cv <- sum((y_cv - y_pred_cv)^2)
  TSS_cv <- sum((y_cv-mean(y_cv))^2)
  r2_cv <- 1-(RSS_cv/TSS_cv)
  df_pred$r2_year_cv <- r2_cv
  
  if (is.null(all_predictions)){
    all_predictions <- df_pred
  } else {
    all_predictions <- rbind(all_predictions, df_pred)
  }
  

}


y_cv <- all_predictions[["y"]]
y_pred_cv <- all_predictions[["y_pred"]]
RSS_cv <- sum((y_cv - y_pred_cv)^2)
TSS_cv <- sum((y_cv-mean(y_cv))^2)
r2_cv <- 1-(RSS_cv/TSS_cv)


unique(all_predictions %>%select(year, r2_year_cv))


  


# create random 10-fold split
k <- 10

fold_id <- sample(rep(1:k, length.out = nrow(data_yield)))
data_yield$fold_id <- fold_id

all_predictions2 <- NULL

for (fold in 1:k) {
  
  train_data <- data_yield %>% filter(fold_id != fold)
  test_data  <- data_yield %>% filter(fold_id == fold)
  
  rf_cv <- ranger(
    formula = as.formula(
      paste(target, "~", paste(predictors, collapse = " + "))
    ),
    data = train_data %>% select(-fold_id),
    num.trees = 1000,
    mtry = floor(sqrt(length(predictors))),
    seed = 42,
    importance = "permutation"
  )
  
  preds <- predict(
    rf_cv,
    data = test_data %>% select(-fold_id)
  )$predictions
  
  df_pred <- data.frame(
    nuts_id = test_data$nuts_id,
    year = test_data$year,
    fold = fold,
    y = test_data[[target]],
    y_pred = preds
  )
  
  if (is.null(all_predictions2)) {
    all_predictions2 <- df_pred
  } else {
    all_predictions2 <- rbind(all_predictions2, df_pred)
  }
}

# overall CV metrics across all held-out predictions
y_cv2 <- all_predictions2[["y"]]
y_pred_cv2 <- all_predictions2[["y_pred"]]

RSS_cv2 <- sum((y_cv2 - y_pred_cv2)^2)
TSS_cv2 <- sum((y_cv2 - mean(y_cv2))^2)
r2_cv2 <- 1 - (RSS_cv2 / TSS_cv2)




cat("10-fold CV R²:", r2_cv2, "\n")

cat("OOB R²:", oob_r2, "\n")

cat("Year-CV R²:", r2_cv, "\n")






# create random 10-fold split
k <- 10
fold_id <- sample(rep(1:k, length.out = nrow(data_yield)))
data_yield$fold_id <- fold_id

data_yield$year2 <- factor(data_yield$year)
all_predictions3 <- NULL
importance_cv <- list()
for (fold in 1:k) {
  
  train_data <- data_yield %>% filter(fold_id != fold)
  test_data  <- data_yield %>% filter(fold_id == fold)
  
  rf_cv <- ranger(
    formula = as.formula(
      paste(target, "~", paste(c(predictors, "year2"), collapse = " + "))
    ),
    data = train_data %>% select(-fold_id),
    num.trees = 1000,
    mtry = floor(sqrt(length(predictors))),
    seed = 42,
    importance = "permutation"
  )
  importance_cv[[fold]] <- importance(rf_cv)
  
  preds <- predict(
    rf_cv,
    data = test_data %>% select(-fold_id)
  )$predictions
  
  df_pred <- data.frame(
    nuts_id = test_data$nuts_id,
    year = test_data$year,
    fold = fold,
    y = test_data[[target]],
    y_pred = preds
  )
  
  if (is.null(all_predictions3)) {
    all_predictions3 <- df_pred
  } else {
    all_predictions3 <- rbind(all_predictions3, df_pred)
  }
}
imp_matrix <- do.call(rbind, importance_cv)

mean_importance <- colMeans(imp_matrix)

sort(mean_importance, decreasing = TRUE)

# overall CV metrics across all held-out predictions
y_cv3 <- all_predictions3[["y"]]
y_pred_cv3 <- all_predictions3[["y_pred"]]

RSS_cv3 <- sum((y_cv3 - y_pred_cv3)^2)
TSS_cv3 <- sum((y_cv3 - mean(y_cv3))^2)
r2_cv3 <- 1 - (RSS_cv3 / TSS_cv3)



cat("OOB R²:", oob_r2, "\n")

cat("Year-CV R²:", r2_cv, "\n")


cat("10-fold CV R²:", r2_cv2, "\n")

cat("10-fold CV R² with year-fixed effect:", r2_cv3, "\n")






# create random 10-fold split
k <- 10
fold_id <- sample(rep(1:k, length.out = nrow(data_yield)))
data_yield$fold_id <- fold_id

data_yield$year2 <- factor(data_yield$year)
all_predictions4 <- NULL

for (fold in 1:k) {
  
  train_data <- data_yield %>% filter(fold_id != fold)
  test_data  <- data_yield %>% filter(fold_id == fold)
  
  rf_cv <- ranger(
    formula = as.formula(
      paste(target, "~", "year2")
    ),
    data = train_data %>% select(-fold_id),
    num.trees = 1000,
    seed = 42,
    importance = "permutation"
  )
  
  preds <- predict(
    rf_cv,
    data = test_data %>% select(-fold_id)
  )$predictions
  
  df_pred <- data.frame(
    nuts_id = test_data$nuts_id,
    year = test_data$year,
    fold = fold,
    y = test_data[[target]],
    y_pred = preds
  )
  
  if (is.null(all_predictions4)) {
    all_predictions4 <- df_pred
  } else {
    all_predictions4 <- rbind(all_predictions4, df_pred)
  }
}

# overall CV metrics across all held-out predictions
y_cv4 <- all_predictions4[["y"]]
y_pred_cv4 <- all_predictions4[["y_pred"]]

RSS_cv4 <- sum((y_cv4 - y_pred_cv4)^2)
TSS_cv4 <- sum((y_cv4 - mean(y_cv4))^2)
r2_cv4 <- 1 - (RSS_cv4 / TSS_cv4)


