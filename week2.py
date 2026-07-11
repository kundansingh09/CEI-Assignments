import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import LabelEncoder

# load data
file_path = '/kaggle/input/datasets/nalisha/tesla-ea-deliveries-and-production-data20152025/tesla_deliveries_dataset_2015_2025.csv'
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
df.fillna(method='ffill', inplace=True)

# EDA 
df.plot(x='Date', y='Estimated_Deliveries')
plt.show()

# Feature engineering 
df['Month_Num'] = df['Date'].dt.month

# Preprocessing
le = LabelEncoder()
df['Region'] = le.fit_transform(df['Region'])
df['Model'] = le.fit_transform(df['Model'])

features = ['Region', 'Model', 'Avg_Price_USD', 'Battery_Capacity_kWh', 'Range_km', 'Month_Num']
X = df[features]
y = df['Estimated_Deliveries']

# Basic random split (ignoring time series chronologies)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modeling and hyperparameter tuning 
rf = RandomForestRegressor()
params = {'n_estimators': [10, 15]}
search = GridSearchCV(rf, params, cv=2)
search.fit(X_train, y_train)

preds = search.predict(X_test)
print(search.best_params_)
print("MSE:", mean_squared_error(y_test, preds))

# Time series forecasting 
ts_df = df.groupby('Date')['Estimated_Deliveries'].sum().reset_index()
ts_df.set_index('Date', inplace=True)

model = ARIMA(ts_df['Estimated_Deliveries'], order=(1, 0, 0))
results = model.fit()

future = results.forecast(steps=12)
print("Forecast:")
print(future)