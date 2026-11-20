# Turbo.az---Car-Price-Prediction
# Turbo.az Car Price Prediction

Predicting used car prices in Azerbaijan from Turbo.az listings.

Turbo.az is the biggest car marketplace in the country. Prices there are all over the place, and it's hard to tell if a car is priced fairly. I wanted to see how well a model could estimate a car's price from the things people actually look at: brand, model, year, mileage, engine and condition.

## Data

- Around 654K raw listings scraped from Turbo.az (56 columns)
- After cleaning and removing duplicates: **~213K listings**
- Target: price in AZN (USD and EUR prices converted at 1.70 and 1.96)

The dataset isn't in this repo because it's too large. Put `cars.csv` in the same folder as the notebook to run it.

## What the notebook does

**Cleaning**
- Dropped columns that don't help with pricing (URLs, phone numbers, VIN, descriptions, etc.)
- Dropped rows with no model or condition, and removed duplicates
- Converted "4 və daha çox" owners and "8+" seats to numbers, and kept a flag for missing values instead of just filling them
- Grouped cities with fewer than 50 listings into "Digər"

**Feature engineering**
- Pulled horsepower, fuel type and engine volume out of strings like `1.6 L/115 a.g./Dizel`
- `age`: years since production
- `km_per_year`: mileage divided by age
- `power_per_liter`, `age_power`, `kilometrage_log`, and an `is_zero_kilometrage` flag for new cars

**Modeling**
- Log-transformed the price, since it's heavily skewed
- One-hot encoding for most categorical columns and target encoding for `Model` (2,400+ unique values)
- 80/20 train/test split, then compared 9 regression models

## Results

| Model | Test R² (log) | Test R² (AZN) | Test MAE (AZN) |
|---|---|---|---|
| **Random Forest** | **0.961** | **0.952** | **2,366** |
| Extra Trees | 0.960 | 0.944 | 2,414 |
| LightGBM | 0.956 | 0.945 | 2,865 |
| XGBoost | 0.955 | 0.943 | 2,954 |
| Hist Gradient Boosting | 0.954 | 0.940 | 2,990 |
| Gradient Boosting | 0.914 | 0.835 | 4,775 |
| Ridge | 0.879 | 0.752 | 5,745 |
| Linear Regression | 0.879 | 0.752 | 5,745 |
| Elastic Net | 0.865 | 0.696 | 6,112 |

Tree-based models beat the linear ones by a wide margin, which makes sense: price doesn't drop in a straight line with age or mileage.

For example, a 2008 Hyundai i30 (diesel, 270K km) listed at 15,000 AZN gets a prediction of about 14,160 AZN.

## Run it yourself

```bash
pip install pandas numpy matplotlib scikit-learn xgboost lightgbm joblib
python -m notebook
```

Needs scikit-learn 1.4 or newer, because of `TargetEncoder`. Trained models are saved to the `models/` folder.

## What I'd improve next

- The data was scraped more than once, so the same car can show up in both train and test. Splitting by listing ID would give a more honest score.
- A few rows have impossible production years, which shows up as outliers in `age`. They should be filtered out.
- Hyperparameter tuning with cross-validation instead of hand-picked values.
- A small Streamlit app where you type in a car's details and get a price.

## Tech

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, Matplotlib
