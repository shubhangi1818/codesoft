# CODESOFT
titanic project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df=pd.read_csv("Titanic-Dataset (1).csv")

df.head()

df.info()

df.describe()

df['Survived'].value_counts()

sns.countplot(x=df['Survived'],hue=df['Pclass'])

df["Sex"]

sns.countplot(x=df['Sex'],hue=df['Survived'])

# Survival rate by sex
df.groupby('Sex')[['Survived']].mean()


df['Sex'].unique()

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()

df['Sex']=labelencoder.fit_transform(df['Sex'])

df.head()

df['Sex'],df['Survived']

sns.countplot(x=df['Sex'],hue=df['Survived'])

df.isna().sum()

df=df.drop(['Age'],axis=1)

df_final=df
df_final.head(10)

#MODEL TRAINING
X=df[['Pclass','Sex']]
Y=df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
log=LogisticRegression(random_state=0)
log.fit(X_train,y_train)

#model prediction
pred=print(log.predict(X_test))

print(y_test)

import warnings
warnings.filterwarnings("ignore")
res=log.predict([[2,0]])
if (res==0):
    print("so sorry!Not Survived")
else:
    print("Survived")


