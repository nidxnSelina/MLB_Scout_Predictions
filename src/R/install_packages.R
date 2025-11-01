packages <- c(
  "readr",     # to read csv
  "dplyr",     # data wrangling
  "caret",     # modeling helper
  "rpart"      # simple model, or use glm
)

install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

invisible(lapply(packages, install_if_missing))
