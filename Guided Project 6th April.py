#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pima_df=pd.read_csv(r'C:\Users\Ashington\OneDrive\Desktop\diabetes.csv')
pima_df.head()
pima_df.tail()
pima_df.describe()
pima_df.info()
pima_df.rename(columns={'Pregnancies':'Preg','BloodPressure':'BP', 'SkinThickness':'Skin','DiabetesPedigreeFunction':'DPF', 'Outcome':'Class'}, inplace= True)
pima_df.head()
pima_df.shape
pima_df.info()
bmi_df=pima_df['BMI']
bmi_df.describe()
sns.kdeplot(bmi_df, bw=2)
plt.xlabel('BMI')
plt.ylabel('Density')
plt.title('BMI Distribution')
plt.show()
pima_df['Class'].value_counts()
#Library for resampling
from sklearn.utils import resample
df_majority = pima_df[pima_df.Class==0]
df_minority = pima_df[pima_df.Class==1]
#Upsampling
df_minority_upsampled=resample(df_minority, replace = True, n_samples=500, random_state=123)

#combine majority and upsampled minority classes
pima_df_upsampled=pd.concat([df_majority, df_minority_upsampled])
#Display new classs counts
pima_df_upsampled.Class.value_counts()
#Downsampling the majority Class
df_majority_downsampled = resample(df_majority, replace = True, n_samples=268, random_state=123)#combine majority and downsampled minority classes
pima_df_downsampled=pd.concat([df_majority_downsampled, df_minority])
#Display new classs counts
pima_df_downsampled.Class.value_counts()
#Function to plot Classs Distribution
def visualize_classes(df_name):
    labels, counts = np.unique(df_name['Class'], return_counts = True)
    colors= ['r', 'b']
    plt.figure(figsize=(8,8))
    plt.bar(labels, counts, color=colors)
    plt.gca().set_xticks(labels)
    plt.xlabel('Class', fontsize= 15)
    plt.ylabel('Count', fontsize= 15)
    plt.title('Class Distribution')
    plt.show()
visualize_classes(pima_df)
visualize_classes(pima_df_upsampled)
visualize_classes(pima_df_downsampled)
from sklearn.decomposition import PCA
pca = PCA().fit(pima_df)
plt.figure(figsize=(8,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components', fontsize = 15)
plt.ylabel('Cumulative Exaplained Variance', fontsize = 15)
plt.title('Selecting the number of important componets', fontsize = 15)
plt.grid()
plt.show()
#Separate data into features and labels/targets
features = pima_df_upsampled.drop('Class', 1)
labels = pima_df_upsampled['Class']
labels.describe()
features.describe()
#SPlit Data
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state = 0)
#STandard Scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, Binarizer
scale_features_std = StandardScaler()
features_train_std = scale_features_std.fit_transform(features_train)
features_test_std = scale_features_std.fit_transform(features_test)
#Verify Standardization
print(features_train_std)
#Robust Scaler
scale_features_rs = RobustScaler()
features_train_rs = scale_features_rs.fit_transform(features_train)
features_test_rs = scale_features_rs.fit_transform(features_test)
#Verify Standardization
print(features_train_rs)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression(solver='lbfgs', max_iter = 200)
logreg.fit(features_train, labels_train)
predictions = logreg.predict(features_test)
print('Logistic Regression Model Accuracy:', accuracy_score(labels_test, predictions))
from pickle import dump
from pickle import load
model = LogisticRegression(max_iter = 200)
model.fit(features_train, labels_train)
filename = 'PIMA_Indians.sav'
dump(model, open(filename, 'wb'))
saved_model = load(open(filename, 'rb'))
import matplotlib
import sklearn
print('Numpy version:', np.__version__)
print('Pandas version:', pd.__version__)
print('MatPlotLib version:', matplotlib.__version__)
print('Sklearn version:', sklearn.__version__)
print('Seaborn version:', sns.__version__)
from platform import python_version
print('Python Version:', python_version())


# In[ ]:




