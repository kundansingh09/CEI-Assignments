# importing needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore') # ignore annoying warnings

# load data
# check the exact path on the right side of kaggle notebook just in case
file_path = '/kaggle/input/datasets/nalisha/tesla-ea-deliveries-and-production-data20152025/tesla_deliveries_dataset_2015_2025.csv'
df = pd.read_csv(file_path)

# fix the date stuff so time series works properly later
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
df = df.sort_values(by='Date')

# forward fill any missing data so we don't break the model
df.fillna(method='ffill', inplace=True)
df.head()

# basic eda - plotting deliveries and checking correlations
plt.figure(figsize=(15, 5))

# trend plot
plt.subplot(1, 2, 1)
sns.lineplot(data=df, x='Date', y='Estimated_Deliveries', hue='Region', ci=None)
plt.title('Deliveries over time')
plt.xticks(rotation=45)

# correlation matrix
plt.subplot(1, 2, 2)
num_cols = df.select_dtypes(include=[np.number]) # only correlate numeric columns
sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation')

plt.tight_layout()
plt.show()

# feature engineering - creating lag features to give the model historical context
df['Lag1'] = df.groupby(['Region', 'Model'])['Estimated_Deliveries'].shift(1)
df['Lag2'] = df.groupby(['Region', 'Model'])['Estimated_Deliveries'].shift(2)

# drop the nans we just made with the shift function
df_ml = df.dropna().copy()

features = ['Region', 'Model', 'Avg_Price_USD', 'Battery_Capacity_kWh', 'Range_km', 'Lag1', 'Lag2']
X = df_ml[features]
y = df_ml['Estimated_Deliveries']

# chronological train test split (not random because it's time series data)
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# setting up the preprocessing and model pipeline
num_features = ['Avg_Price_USD', 'Battery_Capacity_kWh', 'Range_km', 'Lag1', 'Lag2']
cat_features = ['Region', 'Model']

prep = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

pipe = Pipeline([
    ('prep', prep),
    ('rf', RandomForestRegressor(random_state=42))
])

# hyperparameter tuning setup
params = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5]
}

print("training model... this might take a sec")
# using randomized search to save time over grid search
search = RandomizedSearchCV(pipe, params, cv=3, scoring='neg_mean_squared_error', n_iter=5, random_state=42)
search.fit(X_train, y_train)

best = search.best_estimator_
preds = best.predict(X_test)

print(f"best params found: {search.best_params_}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")

# cell 6 update: switching from basic arima to seasonal sarimax
from statsmodels.tsa.statespace.sarimax import SARIMAX

# grouping by date to get total global deliveries for the TS model
ts_df = df.groupby('Date')['Estimated_Deliveries'].sum().reset_index()
ts_df.set_index('Date', inplace=True)

# use SARIMAX to explicitly inject a 12-month seasonal cycle
# order=(p,d,q), seasonal_order=(P,D,Q,s)
model = SARIMAX(ts_df['Estimated_Deliveries'], 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit(disp=False)

# forecast next 12 months
future = results.forecast(steps=12)
future_dates = pd.date_range(start=ts_df.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')

# plot the updated seasonal forecast
plt.figure(figsize=(10, 5))
plt.plot(ts_df.index, ts_df['Estimated_Deliveries'], label='Actual Data')
plt.plot(future_dates, future.values, label='Seasonal Forecast', color='red', linestyle='--')
plt.title('SARIMA Seasonal Forecast for Deliveries')
plt.legend()
plt.grid(True)
plt.show()