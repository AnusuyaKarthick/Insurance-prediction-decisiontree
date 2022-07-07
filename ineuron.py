# -*- coding: utf-8 -*-
"""
Created on Tue May 24 23:04:39 2022

@author: KarthickAnu
"""

#libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
from sklearn import metrics
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

#load the data
data =pd.read_csv(r'D:\Anu\project\deploy ineuron\insurance.csv')
pd.set_option('display.max_columns',None)
df=data
df.head()
df.tail()

#check the shape
df.shape #1338 rows and 7 columns
#check the datatype
df.info()# 3 categorical features, 4 numerical features
#check the column names
df.columns
#check the duplicate values:present
duplicate = df.duplicated()
duplicate
sum(duplicate)
#Removing the duplicates
df=df.drop_duplicates()
sum(df.duplicated())
#check the column names
df.columns
#unique counts
df.nunique()
# sex -male and female
#children - 0 to 5 count
#smoker - smoking and not smoking
#region-  Northeast, northwest, southeast, southwest
#Data preprocessing
#show NA Values and doing imputation
df.isnull().sum()# NO missing values
#To find the statistical property of data
df.describe()
#Measures of central tendency
#finding mean and median for Continuous data
df.mean(numeric_only=True) #mean and median are similar for all except expense column - chance of outliers
df.median()
#Finding mode for categorical data
df[["sex","region","smoker"]].mode()
# Measures of Dispersion / Second moment business decision
df.var() # variance
df.std() # standard deviation
range = max(df.expenses) - min(df.expenses) # range
range
### Third moment business decision##Skeweeness-positive skeweness
df.skew() #range -1 to 1 else highly skewed distribution
# Fourth moment business decision##kurtosis- range -1.2 to 1.6
df.kurt() #=3--> normal distrubiton kurtosis, excess kurtosis for noraml distribution is 0 or negative
#>3--> having more peak and fatter tails -leptokurtotic
#between 1 to 3 - having less peak and thin tails --platykurtotic

#outlier detection and treatment
fig, ax = plt.subplots(2,2,figsize =(16,10))
plt.suptitle('Box plot')
sns.boxplot(data=df,x='age',ax = ax[0][0]) #Right skewed, no outliers
sns.boxplot(data=df,x='bmi',ax = ax[0][1])#right skewed, outliers are present
sns.boxplot(data=df,x='expenses',ax = ax[1][0])#right skewed, outliers are present
sns.boxplot(data=df,x='children',ax = ax[1][1])#right skewed, no outliers

'''def return_outliers(data):
    outliers = []
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3-Q1
    Lower_bound = Q1-1.5*IQR
    Upper_bound = Q3+1.5*IQR
    for i in data:
        if i>Upper_bound or i<Lower_bound:
            outliers.append(i)
    return outliers

return_outliers(df['bmi'])
return_outliers(df['expenses'])

#outlier treatment
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['bmi'])

df.bmi = winsor.fit_transform(df[['bmi']])
# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_
# lets see boxplot
sns.boxplot(df.bmi)
# ##observation 
# outlayers are removed 
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['expenses'])

df.expenses = winsor.fit_transform(df[['expenses']])
# lets see boxplot
sns.boxplot(df.expenses)'''

#Exploratory data analysis:Datavisualization
#count plot 1 - categorical features 
#To check the imbalance in data..
fig, ax = plt.subplots(2,2,figsize =(12,10))
plt.suptitle('Countplot on categorical features')
sns.countplot(x= df.sex, ax = ax[0][0]) #count of male is slightly more than female
sns.countplot(x = df.smoker, ax = ax[0][1])# not smoking people are very less 
sns.countplot(x = df.region, ax = ax[1][0])#southeast people count is more
sns.countplot(x = df.children, ax = ax[1][1])# people having five children is less 

#distribution plot and histogram for continuous features
fig, ax = plt.subplots(2,2,figsize =(12,10))
plt.suptitle('distribution')
sns.distplot(df.age,ax = ax[0][0],bins=15) #uniformly distributed
sns.distplot(df.bmi,ax = ax[0][1],bins=10) #normally distributed
sns.distplot(df.expenses,ax = ax[1][0],bins=15) #right skewed
fig.delaxes(ax[1,1]) 

fig, ax = plt.subplots(2,2,figsize =(12,10))
plt.suptitle('distribution')
sns.histplot(data=df, x="bmi", bins=20,ax = ax[0][0],kde=True)#normally distributed
sns.histplot(data=df, x="age", bins=20,ax = ax[0][1],kde=True)#uniformly distributed
sns.histplot(data=df, x="expenses", bins=20,ax = ax[1][0],kde=True)#right skewed
fig.delaxes(ax[1,1])
#Scatter plot
fig, ax = plt.subplots(2,3,figsize =(12,10))
plt.suptitle('Scatter plot')
sns.scatterplot(x="age",y="bmi", data=df,ax = ax[0][0])
sns.scatterplot(x="age",y="expenses", data=df,ax = ax[0][1]) 
sns.scatterplot(x="bmi",y="expenses", data=df,ax = ax[0][2])
sns.scatterplot(x="smoker",y="expenses", data=df,ax = ax[1][0])
sns.scatterplot(x="sex",y="expenses", data=df,ax = ax[1][1])
sns.scatterplot(x="children",y="expenses", data=df,ax = ax[1][2])

sns.scatterplot(x="region",y="expenses", data=df)
#from above observation, bmi, gender, region doesnot have more impact on expenses
#smoking has strong influence on expense
#As age increases expense also increases


sns.set_theme(color_codes=True)
ax = sns.regplot(x="age", y="expenses", data=df)
ax = sns.regplot(x="bmi", y="expenses", data=df)
ax = sns.regplot(x="age", y="bmi", data=df)

#bar plots
sns.barplot(x = df["sex"], y = df["expenses"], hue = "smoker", data= df).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.show()
#From above observation smoker has more expense irrespective of gender

#bar plots
sns.barplot(x = df["sex"], y = df["bmi"], hue = "smoker", data= df).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.show()
#Male with smoking habit have a slight increase in bmi than female

sns.barplot(x = df["children"], y = df["expenses"], hue = "smoker", data= df).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.show()
#From above observation smoker has more expense irrespective of children count
#Expense decrease for people having more than 3 children

#pie plots
df.groupby(['smoker','region']).sum().plot(kind='pie', y='expenses',autopct='%1.0f%%',figsize=(10,10)).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
#People with more smoking habit belonging to southeast has high premium 
#People belonging to northeast has very less expenses

#pie plots
df.groupby(['sex']).mean().plot(kind='pie', y='expenses',autopct='%1.0f%%',figsize=(10,10)).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
#Female has to pay less premium then male

#pie plots
df.groupby(['sex']).mean().plot(kind='pie', y='bmi',autopct='%1.0f%%',figsize=(10,10)).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
#gender doesnot impact bmi

##pie plots
df.groupby(['children']).mean().plot(kind='pie', y='expenses',autopct='%1.0f%%',figsize=(10,10)).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
#children count greater than 3 are paying less premium  

##pie plots
df.groupby(['smoker']).mean().plot(kind='pie', y='expenses',autopct='%1.0f%%',figsize=(10,10)).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
#children count greater than 3 are paying less premium than  

# To Create dummy variables,import Label Encoder
##############
#convert objects(categorical data) into numeric

categories = ['sex', 'region','smoker']

# Encode Categorical Columns
le = LabelEncoder()
df[categories] = df[categories].apply(le.fit_transform)




# Correlation between different variables
corr = df.corr() #smoker is strongly correlated to expense, region is having negative correlation with expense
# Set up the matplotlib plot configuration
f, ax = plt.subplots(figsize=(12, 10))
# Generate a mask for upper traingle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Configure a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap
sns.heatmap(corr, annot=True, mask = mask, cmap=cmap)

## # normalization
'''import sklearn
from sklearn import preprocessing as per
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer'''

# initialize the scaler
#scaler = StandardScaler()

# Apply the transormation
#standard_Data = scaler.fit_transform(df)
#standard_Data1=pd.DataFrame(standard_Data,index=df.index,columns=df.columns)

#scaler=Normalizer().fit(df)
#normalizeData=scaler.transform(df)
#normalizeData1=pd.DataFrame(normalizeData,index=df.index,columns=df.columns)
#print(normalizeData)        

#Setting the value for X and Y
#x = normalizeData1[['age', 'sex', 'bmi','children','smoker','region']]
#y = df['expenses']
#x=standard_Data1[['age', 'sex', 'bmi','children','smoker','region']]
#y=df.expenses

x=df.iloc[:, :-1]
y=df.expenses
#Splitting the dataset

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
#Fitting the Multiple Linear Regression model

mlr = LinearRegression()  
mlr.fit(x_train,y_train)
#Prediction of test set
y_pred_mlr= mlr.predict(x_test)
#Predicted values
print("Prediction for test set: {}".format(y_pred_mlr))

'''#Intercept and Coefficient
print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(zip(x, mlr.coef_))'''

#Actual value and the predicted value
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
mlr_diff.head()

#Model Evaluation

meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
r  = math.sqrt(rootMeanSqErr)
print('R squared: {:.2f}'.format(mlr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)
print(f'     correlation coefficient = {r:.1f}')

#Decision tree regressor

decision_tree_reg = DecisionTreeRegressor()
decision_tree_reg.fit(x_train, y_train)
y_pred4 = decision_tree_reg.predict(x_test)

print("MSE : ",mean_squared_error(y_pred4,y_test))
print("MAE : ",mean_absolute_error(y_pred4,y_test))
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred4))
r  = math.sqrt(rootMeanSqErr)
print('R squared: {:.2f}'.format(decision_tree_reg.score(x,y)*100))
print(f'     correlation coefficient = {r:.1f}')


import pickle

# open the pickle file in writebyte mode
file = open("ineuronmodel.pkl",'wb')
#dump information to that file
pickle.dump(decision_tree_reg, file)
file.close()

# Loading model to compare the results
model = pickle.load(open('ineuronmodel.pkl','rb'))
print(model.predict([[60,0,25.8,0,0,1]]))

y_pred_pickle = model.predict(x_test)
print('R squared: {:.2f}'.format(decision_tree_reg.score(x,y)*100))




