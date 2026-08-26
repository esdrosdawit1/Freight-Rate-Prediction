# Freight Rate Prediction

## Project Overview

This project develops a machine learning solution for predicting freight rates from shipment and market characteristics.

The goal is to accurately estimate freight rates for new loads using information such as origin, destination, distance, equipment type, weight, date, and available market signals.

The project follows an end-to-end machine learning workflow, including exploratory data analysis, feature engineering, model development, hyperparameter tuning, model comparison, validation, model blending, and final prediction generation.

---

## Business Problem

Freight pricing depends on multiple factors, including:

- Pickup and delivery locations
- Shipment distance
- Equipment type
- Load weight
- Date and seasonality
- Market conditions
- Quote-related signals

Accurately predicting freight rates can support more consistent pricing decisions, improve cost estimation, and provide data-driven insight into freight markets.

This project treats freight-rate prediction as a supervised regression problem, using **Root Mean Squared Error (RMSE)** as the primary evaluation metric.

---

## Dataset

The assessment provided several datasets for different stages of the modeling process:

- `train-test.csv` — labeled development data used for model training and evaluation.
- `validation.csv` — validation loads requiring predicted freight rates.
- `december-chart-inputs.csv` — December chart inputs used for the final prediction task.
- `validation-predictions-template.csv` — template defining the required validation prediction format.

The target variable is the observed freight rate, `posted_rate`.

---

## Exploratory Data Analysis

The data was analyzed to understand the characteristics of the freight-rate problem and identify useful patterns for modeling.

The analysis included:

- Target distribution
- Numerical feature distributions
- Missing-value analysis
- Categorical variables
- Relationships between freight rate and shipment characteristics
- Route-level patterns
- Distance and weight relationships
- Temporal patterns
- Market-related signals
- Quote-related signals

The findings from the exploratory analysis were used to guide the feature-engineering and modeling stages.

---

## Feature Engineering

Feature engineering was used to help the models capture nonlinear relationships, temporal patterns, route-specific behavior, and interactions between shipment characteristics.

### Date Features

The original date variable was transformed into:

- Year
- Month
- Day
- Day of week
- Week of year
- Day of year
- Cyclical month features
- Cyclical day-of-week features

### Distance Features

Additional distance-based features included:

- Squared distance
- Cubed distance
- Log-transformed distance
- Square-root distance

### Weight Features

Additional weight-based features included:

- Squared weight
- Log-transformed weight
- Absolute weight

### Distance and Weight Relationships

The following interaction features were created:

- Distance per weight
- Distance × weight

### Geographic Features

Where geographic coordinates were available, additional features were created from:

- Latitude difference
- Absolute latitude difference
- Longitude difference
- Absolute longitude difference

### Route Features

Route-level categorical features included:

- Pickup location
- Delivery location
- Pickup → delivery route
- Pickup × equipment
- Delivery × equipment
- Route × equipment
- Route × month
- Equipment × month

### Market and Quote Features

Additional interaction features were created using available market and quote signals:

- Market index × distance
- Market index × quote signal
- Quote signal × distance

---

## Data Preprocessing

The preprocessing pipeline was designed to keep the development, validation, and December datasets consistent.

Categorical variables were converted to string values and missing categories were assigned a dedicated `__MISSING__` value.

Numeric missing values were replaced using medians calculated **only from the labeled development data**.

Infinite values were also replaced with missing values before imputation.

The final model feature set was restricted to features available across:

1. The labeled development data
2. The validation data
3. The December chart inputs

This prevents the final December predictions from depending on variables that are unavailable at prediction time.

## Model Development

Multiple machine learning models and hyperparameter configurations were evaluated during development.

The primary modeling approach focused on gradient-boosted decision trees, with CatBoost selected as the strongest modeling framework.

CatBoost was particularly useful for this problem because it can directly handle categorical variables while modeling nonlinear relationships and interactions between features.

Model configurations were evaluated using an October holdout dataset, with RMSE used as the primary metric.

---

## Final Model

The strongest development approach was a **50/50 blend of two tuned CatBoost regression models**.

### CatBoost Tuned 1

- Depth: `6`
- Learning rate: `0.03`
- L2 regularization: `5`
- Random strength: `1`
- Bagging temperature: `1`
- Iterations: `723`

### CatBoost Tuned 3

- Depth: `8`
- Learning rate: `0.02`
- L2 regularization: `10`
- Random strength: `0.5`
- Bagging temperature: `1`
- Iterations: `547`

The final prediction is calculated using equal weighting:

Final Prediction =
0.50 × CatBoost Tuned 1
+
0.50 × CatBoost Tuned 3

## Model Performance

The final 50/50 CatBoost blend achieved an October holdout:

**RMSE: $646.51**

This improved upon the original CatBoost baseline:

**Baseline RMSE: $647.10**

The blended approach therefore provided a measurable improvement over the original baseline model.

## Final Retraining

After selecting the winning models, both CatBoost models were retrained using **100% of the labeled development data in `train-test.csv`**.

The final prediction pipeline:

1. Loads the development, validation, and December datasets.
2. Identifies the target variable.
3. Applies the same feature-engineering pipeline to each dataset.
4. Restricts the final feature set to variables available across all three datasets.
5. Handles categorical variables using CatBoost.
6. Calculates numeric imputation values from the labeled development data.
7. Retrains both winning CatBoost models.
8. Generates predictions for the validation dataset.
9. Generates predictions for the December chart inputs.
10. Combines both model predictions using the 50/50 blend.
11. Performs prediction and output-format validation.
12. Saves the final prediction files.

Retraining on the full labeled development dataset allows the final models to learn from all available labeled observations.

---

## Final Predictions

The final pipeline generates two prediction files.

### `validation_predictions.csv`

This file contains:

load_id,predicted_rate

It provides predictions for all 12,000 validation loads and follows the format specified by the provided validation template.

december_chart_inputs.csv

This file preserves the original December chart inputs and adds:

predicted_rate

Predictions are generated for all 31 December dates.

## Output Validation

Before saving the final files, the pipeline performs several checks to reduce the risk of submission errors.

These include:

- Validation prediction count matches the validation dataset
- December prediction count matches the December dataset
- Predictions contain only finite numeric values
- Predictions are positive
- Validation load_id ordering matches the provided template
- Validation output columns match the required format
- December output columns match the required format
- December contains exactly 31 rows
- December contains dates from December 1 through December 31, 2025
- December shipment characteristics match the required inputs
- No missing predicted rates are present
