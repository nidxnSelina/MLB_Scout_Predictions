library(readr)
library(dplyr)

# -------------------- load data --------------------
train_path <- "data/train.csv"
test_path  <- "data/test.csv"

if (!file.exists(train_path)) {
  stop(paste("[ERROR] train file not found at", train_path))
}
if (!file.exists(test_path)) {
  stop(paste("[ERROR] test file not found at", test_path))
}

train_df <- read_csv(train_path, show_col_types = FALSE)
test_df  <- read_csv(test_path, show_col_types = FALSE)

# -------------------- preprocess data --------------------
# drop unusable columns
drop_cols <- c("Name", "Ticket", "Cabin")
train_df <- train_df[, setdiff(names(train_df), drop_cols)]
test_df  <- test_df[, setdiff(names(test_df), drop_cols)]

# fill Age
if ("Age" %in% names(train_df)) {
  age_med <- median(train_df$Age, na.rm = TRUE)
  train_df$Age[is.na(train_df$Age)] <- age_med
  test_df$Age[is.na(test_df$Age)] <- age_med
}

# fill Fare
if ("Fare" %in% names(test_df)) {
  fare_med <- median(train_df$Fare, na.rm = TRUE)
  test_df$Fare[is.na(test_df$Fare)] <- fare_med
}

# encode Sex
train_df$Sex <- ifelse(train_df$Sex == "female", 1, 0)
test_df$Sex  <- ifelse(test_df$Sex  == "female", 1, 0)

# family features
train_df <- train_df %>%
  mutate(FamilySize = SibSp + Parch + 1,
         IsAlone = ifelse(FamilySize == 1, 1, 0))

test_df <- test_df %>%
  mutate(FamilySize = SibSp + Parch + 1,
         IsAlone = ifelse(FamilySize == 1, 1, 0))

# convert Embarked to factor, fill NAs with most common
if ("Embarked" %in% names(train_df)) {
  mode_emb <- names(sort(table(train_df$Embarked), decreasing = TRUE))[1]
  train_df$Embarked[is.na(train_df$Embarked)] <- mode_emb
  test_df$Embarked[is.na(test_df$Embarked)] <- mode_emb
  train_df$Embarked <- factor(train_df$Embarked)
  test_df$Embarked  <- factor(test_df$Embarked, levels = levels(train_df$Embarked))
}

# ---------------- train model ----------------
train_df$Survived <- as.integer(train_df$Survived)

model <- glm(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + FamilySize + IsAlone,
  data = train_df,
  family = binomial(link = "logit")
)

# print accuracy on train set
train_preds <- ifelse(predict(model, newdata = train_df, type="response") >= 0.5, 1, 0)
train_acc <- mean(train_preds == train_df$Survived)
message(paste("[TRAIN ACCURACY] =", round(train_acc,4)))

# ---------------- make predictions ----------------
test_passenger_ids <- test_df$PassengerId
pred_probs <- predict(model, newdata = test_df, type = "response")
pred_labels <- ifelse(pred_probs >= 0.5, 1, 0)

out_df <- data.frame(
  PassengerId = test_passenger_ids,
  Survived = pred_labels
)

# evaluate and print accuracy on test set
gender_path  <- "data/gender_submission.csv"
answers <- read.csv(gender_path)

merged <- merge(out_df, answers, by = "PassengerId")

test_accuracy <- mean(merged$Survived.x == merged$Survived.y)
cat(sprintf("[INFO] Test Accuracy: %.4f\n", test_accuracy))

# save predictions
out_path <- "data/survival_predictions_r.csv"
write_csv(out_df, out_path)
message(paste("Predictions saved to", out_path))