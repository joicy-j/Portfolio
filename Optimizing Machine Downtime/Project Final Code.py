#!/usr/bin/env python
# coding: utf-8

# # OPTIMIZATION OF MACHINE DOWNTIME
# 
# ## DATA PREPROCESSING

# In[64]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[65]:


df = pd.read_csv("Machine Downtime.csv")


# In[66]:


#Checking for missing values
print(df.isnull().sum())


# In[67]:


#Deleting rows with missing value
df= df.dropna()
print(df.isnull().sum())


# In[68]:


#Checking for Duplicates
print("Number of duplicate rows:", df.duplicated().sum())


# In[72]:


#Converting date to correct format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')


# In[70]:


# Encoding Categorical Variables
df['Downtime'] = df['Downtime'].map({'Machine_Failure': 1, 'No_Machine_Failure': 0})


# ## EXPLORATORY DATA ANALYSIS (EDA)

# In[73]:


pd.DataFrame(data= {'Number': df['Downtime'].value_counts(), 
                    'Percent': df['Downtime'].value_counts(normalize=True)})


# In[75]:


unique_assembly_lines = df['Assembly_Line_No'].unique()
num_unique_assembly_lines = len(unique_assembly_lines)
print("Number of unique assembly lines:", num_unique_assembly_lines)
print("Unique assembly lines:", unique_assembly_lines)


# In[76]:


unique_machine_ids = df['Machine_ID'].unique()
num_unique_machine_ids = len(unique_machine_ids)
print("Number of unique machine IDs:", num_unique_machine_ids)
print("Unique machine IDs:", unique_machine_ids)


# In[78]:


assembly_line_counts = df.groupby('Assembly_Line_No')['Downtime'].value_counts().unstack(fill_value=0)
assembly_line_counts['Total'] = assembly_line_counts.sum(axis=1)
assembly_line_counts['Percent_Failure'] = (assembly_line_counts[1] / assembly_line_counts['Total']) * 100
assembly_line_counts['Percent_Non_Failure'] = (assembly_line_counts[0] / assembly_line_counts['Total']) * 100

# Print the results
print("Downtime Counts and Percentages for Each Assembly Line:")
print(assembly_line_counts)


# In[79]:


# Group the data by machine ID and calculate downtime counts and percentages
machine_id_counts = df.groupby('Machine_ID')['Downtime'].value_counts().unstack(fill_value=0)
machine_id_counts['Total'] = machine_id_counts.sum(axis=1)
machine_id_counts['Percent_Failure'] = (machine_id_counts[1] / machine_id_counts['Total']) * 100
machine_id_counts['Percent_Non_Failure'] = (machine_id_counts[0] / machine_id_counts['Total']) * 100

# Print the results
print("Downtime Counts and Percentages for Each Machine ID:")
print(machine_id_counts)


# In[80]:


import seaborn as sns
import matplotlib.pyplot as plt

corr=df.corr()

plt.figure(figsize=(10, 8))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# Highlight cells where correlation value is higher than 0.6 or lower than -0.6
plt.title("Correlation Matrix")
plt.show()
corr


# ## MACHINE LEARNING MODEL

# In[84]:


#encoding categorical cariables
df = pd.get_dummies(df, columns=['Machine_ID', 'Assembly_Line_No'], drop_first=True)


# In[85]:


#Splitting data into Test and Train datasets
X = df.drop(columns=['Date', 'Downtime'])
y = df['Downtime']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[86]:


#Model Selection and training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[87]:


#Model evaluation 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[88]:


#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[89]:


# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # 5-fold cross-validation
                           scoring='accuracy',
                           n_jobs=-1)  # Use all available CPU cores


# In[90]:


# Perform grid search
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_


print("Best Parameters:", best_params)
print("Best Score:", best_score)


# In[91]:


#Evaluate model with best parameters
best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[92]:


feature_importance = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(feature_importance_df)


# In[ ]:




