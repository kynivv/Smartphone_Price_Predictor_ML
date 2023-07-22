import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import explained_variance_score as evs

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, StackingRegressor, HistGradientBoostingRegressor

# Data Import
df = pd.read_csv('mobile_prices_2023.csv')
df = df.drop('Phone Name', axis=1)


# EDA & Preproccesing
print(df.info(), ('=' * 100))

print(df.isnull().sum(), ('=' * 100))
df.dropna(inplace= True)


# Label Encoder
le = LabelEncoder()

for c in df.columns:
    if df[c].dtype == 'object' :
        df[c] = le.fit_transform(df[c])
    
print(df.dtypes, ('=' * 100))


# Train Test Split
features = df.drop('Price in INR', axis= 1)
target = df['Price in INR']

print(features, ('=' * 100), target, ('=' * 100))

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size= 0.24, random_state= 42)


# Model Training
models = [RandomForestRegressor(n_estimators=100, max_features= 100), BaggingRegressor(), HistGradientBoostingRegressor()]

for m in models:
    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(m, (('<>' * 25) * 3),f'\nTrain Accuracy is : {evs(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'\nTest Accuracy is : {evs(Y_test, pred_test)}')