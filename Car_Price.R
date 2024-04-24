# This is a project to predict the price of used cars in the UK using machine learning models.
# There is a Rmd report for this code as well
# author: "Mateus Spencer"
# date: "2024-04-19"

if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if (!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if (!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
if (!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if (!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if (!require(moments)) install.packages("moments", repos = "http://cran.us.r-project.org")
if (!require(purrr)) install.packages("purrr", repos = "http://cran.us.r-project.org")
if (!require(gplots)) install.packages("gplots", repos = "http://cran.us.r-project.org")
if (!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if (!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if (!require(Metrics)) install.packages("Metrics", repos = "http://cran.us.r-project.org")
if (!require(lightgbm)) install.packages("lightgbm", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(ggplot2)
library(caret)
library(dplyr)
library(data.table)
library(GGally)
library(lubridate)
library(gridExtra)
library(moments)
library(purrr)
library(gplots)
library(corrplot)
library(rpart)
library(Metrics)
library(lightgbm)


# Load the data
audi <- read_csv("data/audi.csv", show_col_types = FALSE)
bmw <- read_csv("data/bmw.csv",show_col_types = FALSE)
ford <- read_csv("data/ford.csv", show_col_types = FALSE)
hyundi <- read_csv("data/hyundi.csv", show_col_types = FALSE)
merc <- read_csv("data/merc.csv",show_col_types = FALSE)
skoda <- read_csv("data/skoda.csv",show_col_types = FALSE)
toyota <- read_csv("data/toyota.csv",show_col_types = FALSE)
vauxhall <- read_csv("data/vauxhall.csv",show_col_types = FALSE)
vw <- read_csv("data/vw.csv",show_col_types = FALSE)


# Check if all the dataframes have the same columns
data_frames <- list(audi, bmw, ford, hyundi, merc, skoda, toyota, vauxhall, vw)

first_column_names <- names(data_frames[[1]])

for (i in seq_along(data_frames)) {
  current_column_names <- names(data_frames[[i]])
  
  if (!all(current_column_names %in% first_column_names)) {    
    print(setdiff(current_column_names, first_column_names))
  }
}

hyundi <- rename(hyundi, tax = `tax(£)`)

# Add a column to each dataframe to indicate the make of the car
audi$make <- "Audi"
bmw$make <- "BMW"
ford$make <- "Ford"
hyundi$make <- "Hyundai"
merc$make <- "Mercedes"
skoda$make <- "Skoda"
toyota$make <- "Toyota"
vauxhall$make <- "Vauxhall"
vw$make <- "Volkswagen"

# Combine all the datasets into one
all_cars <- bind_rows(list(audi, bmw, ford, hyundi, merc, skoda, toyota, vauxhall, vw))

# Create holdout set
set.seed(123)
holdoutIndex <- createDataPartition(all_cars$price, p = 0.9, list = FALSE)

holdout_set <- all_cars[-holdoutIndex, ]
holdout_set_X <- holdout_set %>% select(-price)
holdout_set_Y <- holdout_set$price

# variable description
data.frame(
  `Data Type` = sapply(all_cars, function(col) class(col)[1]),
  Description = c(
    "Model of the car",
    "Year of manufacture",
    "Price of the car",
    "Type of transmission",
    "Mileage of the car",
    "Type of fuel used",
    "Tax on the car",
    "Miles per gallon",
    "Engine size",
    "Make of the car"
  ),
  stringsAsFactors = FALSE
)



# remove duplicates
all_cars <- all_cars[!duplicated(all_cars), ]

# turn into factors

all_cars <- mutate(all_cars, model = as.factor(model), transmission = as.factor(transmission), fuelType = as.factor(fuelType), make = as.factor(make))

holdout_set_X <- mutate(holdout_set_X, model = as.factor(model), transmission = as.factor(transmission), fuelType = as.factor(fuelType), make = as.factor(make))




# vizualize numeric data
all_cars %>%
  select(where(is.factor)) %>% 
  gather(variable, value) %>% 
  ggplot() + 
  geom_bar(aes(x=value), width=0.7, fill = "skyblue", color = "black") + 
  facet_wrap(~ variable, scales="free", ncol = 2) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Bar plots of categorical variables")

numeric_data <- all_cars %>% select(where(is.numeric))

plot_hist_box <- function(data, variable) {
  histogram <- ggplot(data, aes_string(variable)) + 
    geom_histogram(fill = "skyblue", color = "black", bins = 30) +
    ggtitle(paste("Histogram of", variable))
  
  boxplot <- ggplot(data, aes_string(variable)) + 
    geom_boxplot(horizontal = TRUE, fill = "skyblue", color = "black") +
    ggtitle(paste("Boxplot of", variable))
  
  plot <- grid.arrange(boxplot, histogram, ncol = 1)
  
  return(plot)
}

plots <- lapply(names(numeric_data), plot_hist_box, data = numeric_data)

do.call(grid.arrange, c(plots, ncol = 2))


# Descriptive statistics
calc_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

select_if(all_cars, is.numeric) %>%
  map_df(function(x) tibble(
    Mean = mean(x, na.rm = TRUE),
    Median = median(x, na.rm = TRUE),
    Mode = calc_mode(x),
    SD = sd(x, na.rm = TRUE),
    Min = min(x, na.rm = TRUE),
    Max = max(x, na.rm = TRUE),
    Skewness = skewness(x, na.rm = TRUE),
    Kurtosis = kurtosis(x, na.rm = TRUE)
  ), .id = "Variable")


all_cars_clean <- all_cars


#### Year

all_cars_clean <- all_cars_clean %>% filter(year <= 2024 & year >= 1990)
plot <- lapply("year", plot_hist_box, data = all_cars_clean)
do.call(grid.arrange, c(plot, ncol = 1))

#### Mileage

all_cars_clean <- all_cars_clean %>% filter(mileage < 150000)
plot <- lapply("mileage", plot_hist_box, data = all_cars_clean)
do.call(grid.arrange, c(plot, ncol = 1))

#### Miles per Gallon (MPG)

all_cars_clean <- all_cars_clean %>% filter(mpg < 100 & mpg > 15)
plot <- lapply("mpg", plot_hist_box, data = all_cars_clean)
do.call(grid.arrange, c(plot, ncol = 1))

#### Tax

all_cars_high_tax <- all_cars_clean %>% filter(tax > 350) %>% select(where(is.numeric))

plots <- lapply(names(all_cars_high_tax), plot_hist_box, data = all_cars_high_tax)
do.call(grid.arrange, c(plots, ncol = 2))


all_cars_clean <- all_cars_clean %>% filter(tax < 350)
rm (all_cars_high_tax)


all_cars_low_tax <- all_cars_clean %>% filter(tax < 100) %>% select(where(is.numeric))
plots <- lapply(names(all_cars_low_tax), plot_hist_box, data = all_cars_low_tax)
do.call(grid.arrange, c(plots, ncol = 2))


all_cars_clean <- all_cars_clean %>% filter(!(tax < 100 & price > 50000))


#### Engine Size


all_cars_clean <- all_cars_clean %>% filter(engineSize > 0)
plot <- lapply("engineSize", plot_hist_box, data = all_cars_clean)
do.call(grid.arrange, c(plot, ncol = 1))


all_cars_clean_big_engine <- all_cars_clean %>% filter(engineSize > 4)%>% select(where(is.numeric))

plots <- lapply(names(all_cars_clean_big_engine), plot_hist_box, data = all_cars_clean_big_engine)
do.call(grid.arrange, c(plots, ncol = 2))


all_cars_clean <- all_cars_clean %>% filter(!(engineSize > 4 & tax > 200))
plot <- lapply("engineSize", plot_hist_box, data = all_cars_clean)
do.call(grid.arrange, c(plot, ncol = 1))


numeric_data <- all_cars_clean %>% select(where(is.numeric))

plots <- lapply(names(numeric_data), plot_hist_box, data = numeric_data)

do.call(grid.arrange, c(plots, ncol = 2))


# Correlation Matrix
plot <- all_cars_clean %>% select(where(is.numeric)) %>% sample_frac(0.1) %>% ggpairs()
plot # takes a few seconds to display


options(repr.plot.width=10, repr.plot.height=8)
corrplot(all_cars_clean %>% select(where(is.numeric)) %>% 
cor(), method="color", addCoef.col = "black")

# Pairplots
plot1 <- ggplot(all_cars_clean, aes(x = year, y = price, color = fuelType)) + geom_point()
plot2 <- ggplot(all_cars_clean, aes(x = mileage, y = price)) + geom_point()
plot3 <- ggplot(all_cars_clean, aes(x = mpg, y = price, color = fuelType)) + geom_point()
plot4 <- ggplot(all_cars_clean, aes(x = engineSize, y = price, color = transmission)) + geom_point()
plot5 <- ggplot(all_cars_clean, aes(x = mpg, y = tax)) + geom_point()
plot6 <- ggplot(all_cars_clean, aes(x = year, y = mileage)) + geom_point()

grid.arrange(plot1, plot2, plot3, plot4, plot5, plot6, ncol = 2)


# Outliers
all_cars_clean <- all_cars_clean %>% filter(!(mpg >= 55 & tax > 200))


# Boxplots of categoricals vs price
plot1 <- all_cars_clean %>% select(make, price, transmission) %>% 
  ggplot() + geom_boxplot(aes(x=make, y=price, color=transmission)) + 
  ggtitle("Price per make per transmission")

plot2 <- all_cars_clean %>% select(fuelType, make, price) %>% 
  ggplot() + geom_boxplot(aes(x=make, y=price, color=fuelType)) + 
  ggtitle("Price per make per fuelType")

grid.arrange(plot1, plot2, ncol = 1)


# Encoding Categorical Variables
all_cars_clean_encoded <- all_cars_clean
list_categorical <- c("model", "transmission", "fuelType", "make")

all_cars_clean_encoded[list_categorical] <- lapply(all_cars_clean_encoded[list_categorical], function(x) as.numeric(factor(x)))

holdout_set_X[list_categorical] <- lapply(holdout_set_X[list_categorical], function(x) as.numeric(factor(x)))

head(all_cars_clean_encoded)


# Log Transformation
all_cars_clean_encoded_log_mileage <- all_cars_clean_encoded %>% 
  mutate(mileage = log(mileage))

plot <- lapply("mileage", plot_hist_box, data = all_cars_clean_encoded_log_mileage)

do.call(grid.arrange, c(plot, ncol = 1))


qqplot_price_before <- ggplot(all_cars_clean_encoded, aes(sample = price)) + 
  stat_qq() + 
  stat_qq_line() +
  ggtitle("Q-Q Plot of Price (Before Transformation)")

all_cars_clean_encoded <- all_cars_clean_encoded %>% 
  mutate(price = log(price))

holdout_set_Y <- log(holdout_set_Y)

plot <- lapply("price", plot_hist_box, data = all_cars_clean_encoded)

do.call(grid.arrange, c(plot, ncol = 1))



qqplot_price_after <- ggplot(all_cars_clean_encoded, aes(sample = log(price))) + 
  stat_qq() + 
  stat_qq_line() +
  ggtitle("Q-Q Plot of Price (After Transformation)")


grid.arrange(qqplot_price_before, qqplot_price_after, ncol = 2)


# Modeling
results <- data.frame(
  Model = character(),
  RMSE = numeric(),
  R2 = numeric()
)

# Function to evaluate the models
evaluate_model <- function(predictions, model_name, results, actual) {
  predictions_original_scale <- exp(predictions)
  actual_values_original_scale <- exp(actual)

  metrics <- postResample(pred = predictions_original_scale, obs = actual_values_original_scale)

  results <- rbind(results, data.frame(
    Model = model_name,
    RMSE = metrics['RMSE'],
    R2 = metrics['Rsquared'],
    MAE = metrics['MAE']
  ))
  results
}

# Split the data into training and testing
trainIndex <- createDataPartition(all_cars_clean_encoded$price, p = 0.8, list = FALSE)

trainData <- all_cars_clean_encoded[trainIndex, ]
trainData_X <- trainData %>% select(-price)
trainData_Y <- trainData$price

testData <- all_cars_clean_encoded[-trainIndex, ]
testData_X <- testData %>% select(-price)
testData_Y <- testData$price



# Baseline Linear Regression Model
train_control <- trainControl(method = "cv", number = 10)

lr_model <- train(
  price ~ .,
  data = trainData,
  method = "lm",
  trControl = train_control
)

predictions_lr <- predict(lr_model, newdata = testData)

evaluate_model(predictions_lr, "Baseline Linear Regression", results, testData_Y)


# Elastic Net Model

train_control <- trainControl(method = "cv", number = 10)

tuneGrid <- expand.grid(
  alpha = seq(0, 1, by = 0.1), # mix of L1 and L2 regularization
  lambda = 10^seq(-3, 3, length.out = 10) # regularization strength
)

elasticnet_model <- train(
  price ~ ., 
  data = trainData, 
  method = "glmnet", 
  trControl = train_control, 
  tuneGrid = tuneGrid
)

predictions_elasticnet  <- predict(elasticnet_model, newdata = testData)

evaluate_model(predictions_elasticnet, "Elastic Net", results, testData_Y)


# Gradient Boosting Model - LightGBM
dtrain <- lgb.Dataset(data = as.matrix(trainData_X), label = trainData_Y)

# Set up the parameters for LightGBM
params <- list(
  objective = "regression",
  metric = "rmse",
  learning_rate = 0.01,
  num_leaves = 31,
  max_depth = -1,
  min_data_in_leaf = 20,
  bagging_fraction = 1,
  feature_fraction = 1,
  verbose = -1
)

# Perform cross-validation to find the optimal number of iterations
cv_result <- lgb.cv(
  params = params,
  data = dtrain,
  nrounds = 2000,
  nfold = 5,
  early_stopping_rounds = 10,
  verbose = -1
)

# Train the LightGBM model
lightgbm_model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = cv_result$best_iter,
  verbose = -1
)

predictions_lightgbm <- predict(lightgbm_model, newdata = as.matrix(testData_X))

evaluate_model(predictions_lightgbm, "LightGBM", results, testData_Y)


# Results

predictions_lightgbm_holdout <- predict(lightgbm_model, newdata = as.matrix(holdout_set_X))

evaluate_model(predictions_lightgbm_holdout, "LightGBM - Holdout Set", results, holdout_set_Y)


#### Actual vs Predicted Values Plot

combined_predictions <- data.frame(
  Actual = exp(holdout_set_Y),
  Predicted = exp(predictions_lightgbm_holdout)
)

ggplot(combined_predictions, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "dashed") +
  labs(x = "Actual Values", y = "Predicted Values", title = "Actual vs Predicted Values") +
  theme_minimal()


#### Residuals vs Actual Values Plot


ggplot(combined_predictions, aes(x = Actual, y = Predicted - Actual)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed") +
  labs(x = "Actual Values", y = "Residuals", title = "Residuals vs Actual Values")


#### Variable Importance Plot

importance <- lgb.importance(lightgbm_model)
lgb.plot.importance(importance, top_n = 10)

