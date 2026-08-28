# Freight Rate Prediction

## Project Overview

This project develops a machine learning solution for predicting freight rates from shipment, route, equipment, distance, weight, market, and temporal information.

The goal was to build a robust regression pipeline that could generalize from historical freight data to future shipment dates, including a required set of December predictions.

The project follows an end-to-end machine learning workflow:

**Data Exploration → Feature Engineering → Model Development → Time-Based Validation → Hyperparameter Tuning → Model Blending → Full-Data Retraining → Final Predictions**

---

## Dataset

The project uses three primary datasets:

- `train-test.csv` — labeled development data used for model training and time-based validation.
- `validation.csv` — unlabeled shipment records requiring freight-rate predictions.
- `december-chart-inputs.csv` — December shipment inputs requiring daily freight-rate predictions.

A validation prediction template was also provided:

- `validation-predictions-template.csv`

The December dataset contains **31 daily shipment records**, representing December 1 through December 31, 2025.

---

## Problem Definition

This is a **supervised regression problem**.

The target variable is the historical freight rate, represented by the `posted_rate` column.

The model learns relationships between freight rates and variables such as:

- Pickup location
- Delivery location
- Distance
- Equipment type
- Weight
- Date
- Market information
- Quote-related signals

The objective is to minimize **Root Mean Squared Error (RMSE)**.

---

## Exploratory Data Analysis

The initial analysis focused on understanding:

- Dataset dimensions
- Feature types
- Missing values
- Target distribution
- Numerical feature distributions
- Categorical variables
- Relationships between shipment characteristics and freight rates
- Temporal patterns
- Route-level characteristics

Particular attention was given to variables that could influence freight pricing, including distance, weight, route, equipment type, and market conditions.

---

## Feature Engineering

A feature-engineering pipeline was developed to capture nonlinear relationships and interactions within the freight data.

### Date Features

The date variable was transformed into:

- Year
- Month
- Day
- Day of week
- Week of year
- Day of year
- Cyclical month features
- Cyclical day-of-week features

Cyclical transformations allow the model to represent recurring calendar patterns.

### Distance Features

Additional distance features included:

- Distance squared
- Distance cubed
- Log-transformed distance
- Square-root distance

### Weight Features

Additional weight features included:

- Weight squared
- Log-transformed weight
- Absolute weight

### Distance-Weight Relationships

Interaction features were created to capture relationships between shipment size and travel distance:

- Distance per weight
- Distance × weight

### Geographic Features

Where latitude and longitude information was available, the pipeline created:

- Latitude difference
- Absolute latitude difference
- Longitude difference
- Absolute longitude difference

### Route Features

Pickup and delivery locations were combined to create:

- Route
- Route pair

### Equipment Interactions

Interactions between locations and equipment were created:

- Pickup × equipment
- Delivery × equipment
- Route × equipment

### Route and Time Interactions

Additional categorical interactions included:

- Route × month
- Equipment × month

### Market Interactions

Where available, market and quote-related variables were combined with:

- Distance
- Quote signal
- Market index

These features were designed to help the model capture nonlinear pricing behavior.

---

## Model Development

Several regression models and hyperparameter configurations were evaluated during model development.

The final modeling approach used **CatBoostRegressor**, which was particularly well suited to the dataset because it can directly handle categorical variables while also modeling nonlinear relationships.

Categorical variables were handled natively by CatBoost rather than requiring one-hot encoding.

---

## Time-Based Holdout Validation

Instead of relying on a random train/validation split, the project used an **October holdout** to evaluate how well models could generalize to a later time period.

This approach was chosen because freight pricing is time-dependent and a future-period holdout provides a more realistic assessment of model performance.

Models were compared using **RMSE**, with lower values indicating better performance.

---

## Model Selection

After comparing multiple models and hyperparameter configurations, two CatBoost configurations produced the strongest development performance.

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

---

## Model Blending

Rather than selecting only one of the two strongest models, their predictions were combined using a simple **50/50 ensemble**.

The final prediction was calculated as:

**Final Prediction = 0.50 × CatBoost Tuned 1 + 0.50 × CatBoost Tuned 3**

This combines two models with different tree depths and regularization settings, allowing the final prediction to benefit from both model configurations.

---

## Model Performance

The final 50/50 CatBoost blend achieved an October holdout **RMSE of $646.51**, improving on the original CatBoost baseline of **$647.10**.

| Model | October Holdout RMSE |
|---|---:|
| Original CatBoost Baseline | $647.10 |
| 50/50 Tuned CatBoost Blend | **$646.51** |

The final ensemble improved the October holdout RMSE by **$0.59** compared with the original baseline.

---

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

This file contains `load_id, predicted_rate` pairs. It provides predictions for all 12,000 validation loads and follows the format specified by the provided validation template.

### `december_chart_inputs.csv`

This file preserves the original December chart inputs and adds a `predicted_rate` column. Predictions are generated for all 31 December dates.

---

## Output Validation

Before saving the final files, the pipeline performs several checks to reduce the risk of submission errors. These include:

- Validation prediction count matches the validation dataset
- December prediction count matches the December dataset
- Predictions contain only finite numeric values
- Predictions are positive
- Validation `load_id` ordering matches the provided template
- Validation output columns match the required format
- December output columns match the required format
- December contains exactly 31 rows
- December contains dates from December 1 through December 31, 2025
- December shipment characteristics match the required inputs
- No missing predicted rates are present

---

## Project Structure

```
freight-rate-prediction/
│
├── README.md
│
├── notebooks/
│   └── freight_rate_prediction.ipynb
│
├── outputs/
│   ├── validation_predictions.csv
│   └── december_chart_inputs.csv
│
└── requirements.txt

---

## Technologies Used

- Python
- Pandas
- NumPy
- CatBoost
- Matplotlib
- Jupyter Notebook
- Kaggle

---

## Key Skills Demonstrated

- Exploratory Data Analysis
- Data Cleaning
- Missing-Value Handling
- Feature Engineering
- Categorical Feature Engineering
- Time-Based Feature Engineering
- Nonlinear Feature Transformations
- Interaction Features
- Regression Modeling (CatBoost)
- Hyperparameter Tuning
- Model Comparison
- Model Blending
- Time-Based Holdout Validation
- Final Model Retraining
- Prediction Pipeline Development
- Output Validation

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for freight-rate prediction, from exploratory analysis and feature engineering through model selection, time-based validation, ensemble blending, and final prediction generation.

The final solution uses a 50/50 blend of two tuned CatBoost regression models, achieving an October holdout RMSE of $646.51. The winning models were subsequently retrained on the full labeled development dataset and used to generate the required validation and December freight-rate predictions.

The project demonstrates practical machine learning skills including feature engineering, categorical modeling, hyperparameter tuning, time-based validation, ensemble modeling, and production-oriented prediction validation.
