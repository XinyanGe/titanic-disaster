# Titanic survival prediction with logistic regression (R)
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

data_dir    <- Sys.getenv("DATA_DIR", "../data")
train_path  <- file.path(data_dir, "train.csv")
test_path   <- file.path(data_dir, "test.csv")
labels_path <- file.path(data_dir, "gender_submission.csv")
out_path    <- file.path(data_dir, "predictions_r.csv")

banner <- function(txt) {
  cat("\n", strrep("=", 70), "\n", txt, "\n", strrep("=", 70), "\n", sep = "")
}

extract_title <- function(names) {
  t <- trimws(sub("^.*?,\\s*([^\\.]+)\\..*$", "\\1", names, perl = TRUE))
  t[t %in% c("Mlle", "Ms")] <- "Miss"
  t[t == "Mme"] <- "Mrs"
  t[!t %in% c("Mr", "Mrs", "Miss", "Master")] <- "Rare"
  t
}

fit_fill_values <- function(df) {
  titles <- extract_title(df$Name)
  emb <- df$Embarked[!is.na(df$Embarked) & df$Embarked != ""]
  list(
    age_by_title = tapply(df$Age, titles, median, na.rm = TRUE),
    age_overall  = median(df$Age, na.rm = TRUE),
    fare         = median(df$Fare, na.rm = TRUE),
    embarked     = names(sort(table(emb), decreasing = TRUE))[1]
  )
}

preprocess <- function(df, fills) {
  df <- df %>%
    mutate(Title = extract_title(Name),
           FamilySize = SibSp + Parch + 1,
           IsAlone = as.integer(FamilySize == 1))
  cat("[change] Added 'Title' (from Name), 'FamilySize' (SibSp+Parch+1), 'IsAlone'\n")
  print(table(df$Title))

  n_age <- sum(is.na(df$Age))
  df$Age <- ifelse(is.na(df$Age), unname(fills$age_by_title[df$Title]), df$Age)
  df$Age[is.na(df$Age)] <- fills$age_overall
  cat(sprintf("[change] Filled %d missing Age values with the training median Age per Title\n", n_age))

  n_fare <- sum(is.na(df$Fare))
  df$Fare[is.na(df$Fare)] <- fills$fare
  cat(sprintf("[change] Filled %d missing Fare values with training median %.2f\n", n_fare, fills$fare))

  miss_emb <- is.na(df$Embarked) | df$Embarked == ""
  df$Embarked[miss_emb] <- fills$embarked
  cat(sprintf("[change] Filled %d missing Embarked values with training mode '%s'\n",
              sum(miss_emb), fills$embarked))

  df <- df %>%
    mutate(Sex = as.integer(Sex == "female"),
           Pclass = factor(Pclass, levels = c(1, 2, 3)),
           Embarked = factor(Embarked, levels = c("C", "Q", "S")),
           Title = factor(Title, levels = c("Master", "Miss", "Mr", "Mrs", "Rare"))) %>%
    select(any_of("Survived"), Pclass, Sex, Age, SibSp, Parch, Fare,
           Embarked, Title, FamilySize, IsAlone)
  cat("[change] Encoded Sex: female=1, male=0; Pclass/Embarked/Title as factors\n")
  cat("[change] Dropped unused columns: PassengerId, Name, Ticket, Cabin\n")
  df
}

accuracy <- function(actual, pred) mean(actual == pred)

# ---------------------------------------------------------------------------
banner("STEP 1: Load training data")
if (!file.exists(train_path)) stop(paste(train_path, "not found. See README."))
train <- read_csv(train_path, show_col_types = FALSE)
cat(sprintf("Loaded train.csv with %d rows and %d columns\n", nrow(train), ncol(train)))
print(head(train))

banner("STEP 2: Explore the training data")
cat("Missing values per column:\n")
print(colSums(is.na(train)))
cat(sprintf("\nOverall survival rate: %.3f\n", mean(train$Survived)))
cat("\nSurvival rate by Sex:\n")
print(train %>% group_by(Sex) %>% summarise(rate = round(mean(Survived), 3)))
cat("\nSurvival rate by Pclass:\n")
print(train %>% group_by(Pclass) %>% summarise(rate = round(mean(Survived), 3)))

banner("STEP 3: Clean and engineer features (training set)")
fills <- fit_fill_values(train)
train_clean <- preprocess(train, fills)
cat(sprintf("\nFinal training data: %d rows x %d columns\n", nrow(train_clean), ncol(train_clean)))

banner("STEP 4: Fit logistic regression")
formula <- Survived ~ Pclass + Title + Age + Fare + Embarked + FamilySize + IsAlone
cat("[change] Model formula: Survived ~ Pclass + Title + Age + Fare + Embarked + FamilySize + IsAlone\n")
cat("         (Sex is left out because Title already encodes it; SibSp/Parch are left out because FamilySize = SibSp + Parch + 1)\n")
model <- glm(formula, data = train_clean, family = binomial())
cat("Model fitted. Coefficients:\n")
print(round(coef(model), 3))

banner("STEP 5: Accuracy on the training set")
train_pred <- as.integer(predict(model, train_clean, type = "response") > 0.5)
cat(sprintf("Training accuracy: %.4f\n", accuracy(train_clean$Survived, train_pred)))

set.seed(42)
idx <- sample(seq_len(nrow(train_clean)), size = floor(0.8 * nrow(train_clean)))
val_model <- glm(formula, data = train_clean[idx, ], family = binomial())
val_pred <- as.integer(predict(val_model, train_clean[-idx, ], type = "response") > 0.5)
cat(sprintf("Hold-out validation accuracy (80/20 split of train.csv): %.4f\n",
            accuracy(train_clean$Survived[-idx], val_pred)))

banner("STEP 6: Load test data and predict")
if (!file.exists(test_path)) stop(paste(test_path, "not found. See README."))
test <- read_csv(test_path, show_col_types = FALSE)
cat(sprintf("Loaded test.csv with %d rows and %d columns\n", nrow(test), ncol(test)))
test_clean <- preprocess(test, fills)
test_pred <- as.integer(predict(model, test_clean, type = "response") > 0.5)
cat(sprintf("Predicted survival rate on test set: %.3f\n", mean(test_pred)))
out <- data.frame(PassengerId = test$PassengerId, Survived = test_pred)
write_csv(out, out_path)
cat(sprintf("Saved predictions to %s\n", out_path))

banner("STEP 7: Accuracy on the test set")
if (file.exists(labels_path)) {
  ref <- read_csv(labels_path, show_col_types = FALSE)
  merged <- inner_join(out, ref, by = "PassengerId", suffix = c("_pred", "_ref"))
  cat(sprintf("Accuracy vs gender_submission.csv: %.4f\n",
              accuracy(merged$Survived_ref, merged$Survived_pred)))
  cat("NOTE: Kaggle does not publish the true labels for test.csv. gender_submission.csv is",
      "Kaggle's sample baseline (every female survives), so this number is agreement with",
      "that baseline, not true accuracy. Upload predictions_r.csv to Kaggle for the true score.\n")
} else {
  cat("gender_submission.csv not found, so no test labels are available.",
      "Upload predictions_r.csv to Kaggle to get the true test score.\n")
}

banner("DONE")
