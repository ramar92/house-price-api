import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

data = pd.read_csv("house_price.csv")

X = data[['Area','Bedrooms','Age']]
y = data['Price']

model = RandomForestRegressor()

model.fit(X,y)

joblib.dump(model,"model.pkl")

print("Model Saved")