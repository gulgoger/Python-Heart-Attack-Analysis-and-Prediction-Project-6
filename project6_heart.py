# import libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve

import warnings
warnings.filterwarnings("ignore")

# Read and Analyse Data

df = pd.read_csv("heart.csv")
    
df.head()

describe = df.describe()

df.info()

# Missing Value Analysis
df.isnull().sum()

# Unique Value Analysis
for i in list(df.columns):
    print("{} -- {}".format(i,df[i].value_counts().shape[0]))

# Categorical Feature Analysis
categorical_list = ["sex","cp","fbs","restecg","exng","slp","caa","thall","output"]

df_categoric = df.loc[:,categorical_list]
for i in categorical_list:
    plt.figure()
    sns.countplot(x=i, data=df_categoric, hue="output")
    plt.title(i)

# Numeric Feature Analysis

numeric_list = ["age","trtbps","chol","thalachh","oldpeak","output"]

df_numeric = df.loc[:, numeric_list]
sns.pairplot(df_numeric, hue="output", diag_kind="kde")
plt.show()

#Standardization

scaler = StandardScaler()
scaled_array = scaler.fit_transform(df[numeric_list[:-1]])
scaled_array

#Box Plot Analysis
df_dummy = pd.DataFrame(scaled_array,columns= numeric_list[:-1])
df_dummy.head()


df_dummy = pd.concat([df_dummy, df.loc[:,"output"]],axis=1)
df.head()

data_melted = pd.melt(df_dummy, id_vars="output",var_name="features",value_name ="value")
data_melted.head(20)

plt.figure()
sns.boxplot(x="features", y="value", hue="output",data=data_melted)
plt.show()

# Swarm Plot Analysis
plt.figure()
sns.swarmplot(x="features", y="value", hue="output",data=data_melted)
plt.show()

# Cat Plot Analysis
sns.catplot(x="exng",y="age",hue="output",col="sex",kind="swarm",data =df)
plt.show()

#Correlation Analysis
plt.figure(figsize=(7,5))
sns.heatmap(df.corr(),annot=True, fmt= ".1f", linewidths=.7)
plt.show()

# Outlier Detection
numeric_list = ["age","trtbps","chol","thalachh","oldpeak"]
df_numeric = df.loc[:,numeric_list]
df_numeric.head()

df.describe()

#outlier
for i in numeric_list:
    
    #IQR
    Q1 = np.percentile(df.loc[:,i],25)
    Q3 = np.percentile(df.loc[:,i],75)
    
    IQR = Q3 - Q1
    
    print("Old shape:", df.loc[:,i].shape)
    
    # upper bound
    upper = np.where(df.loc[:,i] >= (Q3 + 2.5+IQR))
    
    # lower bound
    lower = np.where(df.loc[:,i] <= (Q1 - 2.5+IQR))  # where founds index
    
    print("{} -- {}".format(upper,lower))
    
    try:
        df.drop(upper[0],inplace=True)
    except: print("KeyError: {} not found in axis".format(upper[0]))
        
    try:
        df.drop(lower[0],inplace=True)
    except: print("KeyError: {} not found in axis".format(lower[0]))
        
    print("New shape:", df.shape)


# Modeling

df1 = df.copy()

#encoding categorical columns
df1 = pd.get_dummies(df1, columns = categorical_list[:-1], drop_first = True)
df1.head()

X = df1.drop(["output"],axis=1)
y = df1[["output"]]

#scaling numeric
scaler= StandardScaler()

X[numeric_list[:-1]] = scaler.fit_transform(X[numeric_list[:-1]])
X.head()

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=3)
print("X_train: {}".format(X_train.shape))
print("X_test: {}".format(X_test.shape))
print("y_train: {}".format(y_train.shape))
print("y_train: {}".format(y_train.shape))


#Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

#calculate probabilities
y_pred_proba = logreg.predict_proba(X_test)
y_pred_proba

y_pred = np.argmax(y_pred_proba, axis=1) #argmax degerlerin buyugunu alir
y_pred

print("Test accuracy: {}".format(accuracy_score(y_pred,y_test)))

# ROC Curve
fpr,tpr, thresholds = roc_curve(y_test,y_pred_proba[:,1])

#plot curve
plt.plot([0,1],[0,1],"k--")
plt.plot(tpr,fpr,label="Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.show()

#Logistic Regression Hyperparameter Tuning
lr = LogisticRegression()

penalty = ["l1","l2"]
parameters= {"penalty":penalty}

lr_searcher = GridSearchCV(lr,parameters)
lr_searcher.fit(X_train,y_train)
print("Best parameters:",lr_searcher.best_params_)

y_pred = lr_searcher.predict(X_test)
print("Test accuracy: {}".format(accuracy_score(y_pred,y_test)))




















