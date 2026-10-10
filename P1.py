import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
train_set=pd.read_csv(r'C:\Users\Ganesh\OneDrive\Desktop\GANI_DS\PROJECTS\ADVANCE_HOUSE_PRICE_PREDICTION\DATASETS\archive\train.csv')
test_set=pd.read_csv(r'C:\Users\Ganesh\OneDrive\Desktop\GANI_DS\PROJECTS\ADVANCE_HOUSE_PRICE_PREDICTION\DATASETS\archive\test.csv')
Train_set=train_set.copy()
Test_set=test_set.copy()
print(test_set.shape)
print(train_set.shape)
        # DATA ANALYSIS
  # no.of missing values in each feature
a=train_set.isnull().sum()
a=a[a>0]
 # copying train data set into a new data frame
A=train_set.copy()
b=train_set.copy()
for i in b.iloc[:,1:-1].columns:
    b[i]=b[i].isnull().astype(int)  # converting NAN to 1's and non NAN to 0's
c=train_set.iloc[:,1:-1].isnull().sum()
c=c[c>0]
  # Grouping and plotting of feature vs output
for i in c.index.astype(str):
    result=b.groupby(i)['SalePrice'].mean()
    plt.bar(result.index.astype(str),result.values)
    plt.title(i)
    #plt.show()
  # Numerical features
Data=train_set.select_dtypes(include=['number'])
data=Data.copy()
print(Data.shape)
print(Data.head())
print(Data.columns)
  # Discrete features (BIVARIATE ANALYSIS)
for col in Data.iloc[:,1:-1].columns:
    if Data[col].nunique()<20:
     r=Data.groupby(col)['SalePrice'].mean().sort_values()
   # plt.bar(r.index.astype(str),r.values)
   # plt.title(col)
    #plt.show()
  # Continuous features
  # HISTOGRAM PLOT TO CHECK DISTRIBUTION  (UNIVARIATE ANALYSIS)
for col in Data.iloc[:,1:-1].columns:
   if Data[col].nunique()>20:
     plt.hist(Data[col].dropna(),bins=30)
     plt.xlabel(col)
     #plt.show()

# SCATTER PLOT BETWEEN CONTINUOUS FEATURES AND TARGET (BIVARIATE ANALYSIS)
for col in Data.iloc[:,1:-1].columns:
   if Data[col].nunique()>20:
      plt.scatter(Data[col],Data['SalePrice'])
      plt.title(f"{col} vs SalePrice")
      plt.xlabel(col)
      plt.ylabel("SalePrice")
      #plt.show()
	# features Correlation with target (BIVARIATE ANALYSIS)
corr=Data.corr(numeric_only=True)
print(corr['SalePrice'].sort_values(ascending=False))

  # OUTLIERS DETECTION USING BOX PLOT ONLY FOR CONTINOUS FEATURES NOT WORKS FOR CATEGORICAL FEATURES
for col in Data.iloc[:,1:-1].columns:
   if Data[col].nunique()<20:
     Data.boxplot(column=col)
     plt.ylabel(col)
     plt.title(col)
     #plt.show()

   # Categorical variables
cat=train_set.select_dtypes(include='object')
print("\nCategorical:")
for col in cat.columns:
   print(col)
	# Finding categories within categorical features
for col in cat.columns:
   print(f"THE FEATURE IS {col} AND DIFFERENT CATEGORIES IN IT ARE {cat[col].unique()}")
  # Finding out relation between categorical and dependent feature
for col in cat.columns:
   a=train_set.groupby(col)['SalePrice'].mean().sort_values()
   plt.bar(a.index,a.values)
   plt.xlabel(col)
   plt.ylabel('SalePrice')
   plt.title(col)
   #plt.show()
t_data=test_set.select_dtypes(include='number')
t_data=t_data.iloc[:,1:]
                           # FEATURE ENGINEERING
                  # HANDLING CATEGORICAL FEATURES MISSING VALUES
d=cat.isnull().sum()
d=d[d>0]
print(d.index)
               # REPLACING MISSING VALUES WITH A NEW LABEL
train_set=cat=cat.fillna("MISSING")
test_set=test_set.select_dtypes(include='object').fillna("MISSING")
print(test_set.select_dtypes(include='object'))
t_cat=test_set.select_dtypes(include='object')

                   # HANDLING NUMERICAL FEATURES MISSING VALUES
print(data.head())
D=data.isnull().sum()
D=D[D>0]
print(D)
                   # REPLACING THE NUMERICAL MISSING VALUES
            # WHENEVER WE HAVE MORE OUTLIERS, IT'S BETTER TO REPLACE THE MISSING VALUES WITH "MEDIAN" OR "MODE"
for col in data.columns:
   median=data[col].median()
   train_set[col]=data[col]=data[col].fillna(median)
for col in t_data.columns:
   Median=t_data[col].median()
   test_set[col]=t_data[col]=t_data[col].fillna(Median)
            # TEMPORAL VARIABLES (DATE TIME VARIABLES/FEATURES)
for col in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
   train_set[col]=data[col]=data['YrSold']-data[col]
   test_set[col]=t_data[col]=t_data['YrSold']-t_data[col]
print(data[['YearBuilt','YearRemodAdd','GarageYrBlt']].head())
print(t_data[['YearBuilt','YearRemodAdd','GarageYrBlt']].head())

               # CONVERSION OF SKEWED DISTRIBUTIONS INTO NORMAL DISTRIBUTIONS
for col in data.select_dtypes(include=['float64','int64']).columns:
   if skew(data[col]) > 0.5:
      train_set[col]=np.log1p(data[col])
for col in t_data.columns:
   if skew(t_data[col]) > 0.5:
      test_set[col]=np.log1p(t_data[col])

               # HANDLING RARE CATEGORICAL FEATURES
for col in cat.columns:
   if col != 'SalePrice':
    temp=train_set.groupby(col)['SalePrice'].count()/len(train_set)
    temp_df=temp[temp>0.01].index
    rare_var=temp[temp<0.01].index
    train_set[col]=cat[col]=np.where(train_set[col].isin(temp_df),train_set[col],'Rare_var')
for col in t_cat.columns:
   test_set[col]=t_cat[col]=test_set[col].apply(lambda x: 'Rare_var' if x in rare_var
                                else x)

              # FEATURE ENCODING OF CATEGORICAL FEATURES
Cat=train_set.select_dtypes(include='object')
Cat=Cat.iloc[:,:-1]
combined=pd.concat([Cat,t_cat],axis=0)
ordinal_cols = [
    'ExterQual', 'ExterCond',
    'BsmtQual', 'BsmtCond',
    'HeatingQC', 'KitchenQual'
]

quality_map = {
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5,
    'MISSING': 0,
}

for col in ordinal_cols:
    combined[col] = combined[col].map(quality_map)
nominal_cols=combined.columns.drop(ordinal_cols)
combined=pd.get_dummies(combined,columns=nominal_cols,drop_first=True)
n_train=train_set.shape[0]
train_split=combined.iloc[:n_train,:]
test_split=combined.iloc[n_train:,:]
train_final=pd.concat([train_split,train_set.select_dtypes(include=['int64','float64'])],axis=1)
train_final=pd.concat([Train_set['Id'],train_final],axis=1)

test_final=pd.concat([test_split,test_set.select_dtypes(include=['int64','float64'])],axis=1)
test_final=pd.concat([Test_set['Id'],test_final],axis=1)
data.to_csv('final_train.csv',index=False)
data.to_csv('final_test.csv',index=False)
                                                 # FEATURE SELECTION
para={
   "alpha":[0.01,0.001,0.1,1,10,100]
}
x_train_all=train_final.drop(['Id','SalePrice'],axis=1).fillna(0)
y_train_all=train_final['SalePrice']

# Split data for accuracy evaluation (80% train, 20% validation)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)
x_test=test_final.drop(['Id'],axis=1).fillna(0)

# Align columns
x_train, x_test=x_train.align(x_test,join='left',axis=1,fill_value=0)
x_val, x_test=x_val.align(x_test,join='left',axis=1,fill_value=0)

scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_val=scaler.transform(x_val)
x_test=scaler.transform(x_test)

lasso=Lasso(max_iter=10000)
grid=GridSearchCV(lasso,para,cv=5)
grid.fit(x_train,y_train)
best_lasso=grid.best_estimator_

# Accuracy evaluation on validation set
y_val_pred_log=best_lasso.predict(x_val)
y_val_pred=np.expm1(y_val_pred_log)
y_val_actual=np.expm1(y_val)
print("\n--- Model Accuracy (Validation Set) ---")
print(f"MAE: {mean_absolute_error(y_val_actual, y_val_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val_actual, y_val_pred)):.2f}")
print(f"R2 Score: {r2_score(y_val_actual, y_val_pred):.4f}")

selector=SelectFromModel(best_lasso,prefit=True)
x_train_selected=selector.transform(x_train)
x_test_selected=selector.transform(x_test)
best_lasso.fit(x_train_selected,y_train)
y_pred_log=best_lasso.predict(x_test_selected)
y_pred_log=best_lasso.predict(x_test_selected)
y_pred=np.expm1(y_pred_log)
submission=pd.DataFrame({
   "Id":test_final['Id'],
   "SalePrice":y_pred
})

print(submission.head())
print(y_pred_log.min(),y_pred_log.max())
print(y_pred.min(),y_pred.max())
submission.to_csv('PREDICTED_SALEPRICES.csv',index=False)
