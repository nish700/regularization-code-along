# --------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.linear_model import Ridge, Lasso


## Load the data
data = pd.read_csv(path)

## Split the data and preprocess
print(data['source'].value_counts())

train = data[data['source']=='train'].drop('source',axis=1)
test = data[data['source']=='test'].drop('source',axis=1)

## Baseline regression model

train_base = train[['Item_Weight','Item_MRP','Item_Visibility']]

X = train_base
y = train['Item_Outlet_Sales']

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2,random_state=0)

# print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

linear_reg = LinearRegression()

linear_reg.fit(X_train, y_train)

y_pred = linear_reg.predict(X_val)

mse = mean_squared_error(y_pred, y_val)

r2_score_basic = r2_score(y_val, y_pred)

print("Baseline regression model.","mean squared error is: ",mse,"r2 score is: " ,r2_score_basic)

## Effect on R-square if you increase the number of predictors

train_new = train.drop(['Item_Outlet_Sales','Item_Identifier'],axis=1)

X_new = train_new
y_new = y

X1_train, X1_test , y1_train, y1_test = train_test_split(X_new,y_new, test_size= 0.2, random_state=3)

linear_reg.fit(X1_train,y1_train)

y1_pred = linear_reg.predict(X1_test)

mse_1 = mean_squared_error(y1_pred,y1_test)

# print(mse_1)
# print(y1_pred.shape,y1_val.shape,'abc')

r2_score_1 = r2_score(y1_test, y1_pred)

print('Effect of increase the number of predictors:',"mean squared error:",mse_1,"R2 score:",r2_score_1)

## Effect of decreasing feature from the previous model
train_reduced = train.drop(['Item_Outlet_Sales','Item_Identifier', 'Item_Visibility', 'Outlet_Years'],axis=1)

X_reduced = train_reduced
y_reduced = y

X2_train, X2_test, y2_train, y2_test = train_test_split(X_reduced,y_reduced,test_size=0.2,random_state=2)

linear_reg.fit(X2_train,y2_train)

y2_pred = linear_reg.predict(X2_test)

mse_2 = mean_squared_error(y2_pred,y2_test)

r2_score_2 = r2_score(y2_test,y2_pred)

print('Effect of decreasing feature from the previous model.', 'mean squared error:',mse_2,' r2 score:',r2_score_2)

## Detecting hetroskedacity


## Model coefficients


## applying Ridge regression for all the features
ridge = Ridge()
ridge.fit(X1_train,y1_train)
y_pred_ridge = ridge.predict(X1_test)

rmse_ridge = np.sqrt(mean_squared_error(y_pred_ridge,y1_test))
print("RMSE values for predictions done by ridge model:",rmse_ridge)

## Lasso regression
lasso = Lasso()
lasso.fit(X1_train, y1_train)
y_pred_lasso = lasso.predict(X1_test)
rmse_lasso = np.sqrt(mean_squared_error(y_pred_lasso,y1_test))
print("RMSE values for predictions done by Lasso Model:", rmse_lasso)

model = "Lasso" if rmse_lasso < rmse_ridge else "Ridge"

print(model, 'has lower rmse value and better fits the data')

## Cross vallidation
rmse_L1 = np.mean(cross_val_score(lasso,X1_train,y1_train,cv=10))
rmse_L2 = np.mean(cross_val_score(ridge, X1_train, y1_train, cv=10))

model_2 = lasso if rmse_L1 < rmse_L2 else ridge

model_2.fit(X1_train,y1_train)

pred = model_2.predict(X1_test)

error = np.sqrt(mean_squared_error(pred,y1_test))

print('residual error is ', error)



