#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
import pandas as pd

df = pd.read_excel('life expectancy.xlsx')
df


# In[5]:


df.isnull().sum()


# In[8]:


df_cleaned = df.dropna(axis=0)
df_cleaned


# In[9]:


df_renamed = df_cleaned.rename(columns={"Life expectancy at birth, total (years)": "Life expectancy"})
df_renamed


# In[10]:


import matplotlib.pyplot as plt

# Data
x = df_renamed['Year']
y = df_renamed['Life expectancy']
# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', color='b', linestyle='-', label='Life expectancy Rate')
plt.title('Life expectancy Rate in Germany')
plt.xlabel('Year')
plt.ylabel('Life expectancy(%)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


# In[11]:


# Data
x = df_renamed['Year']
y = df_renamed['Life expectancy']

# Create the plot
plt.figure(figsize=(10, 6))
plt.bar(x, y, color='g', linestyle='-', label='Life expectancy')
plt.title('Life expectancy Rate in Germany (1992-2023)')
plt.xlabel('Year')
plt.ylabel('Life expectancy Rate (%)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


# In[12]:


import seaborn as sns
sns.pairplot(df_renamed[['Year', 'Life expectancy']])


# In[50]:


import numpy as np
from sklearn.linear_model import LinearRegression
# Split data into features (X) and target (y)
X_fit = df_renamed[['Year']]
Y_fit = df_renamed['Life expectancy']

# Initialize and fit a linear regression model
model = LinearRegression()
model.fit(X_fit, Y_fit)

# Predict life expectancy for all years
all_years = range(1960, 2022)  # Adjust the range as needed
y_pred = model.predict(np.array(all_years).reshape(-1, 1))

# Create a new DataFrame with predictions
predictions_df = pd.DataFrame({'Year': all_years, 'Predicted_Life_Expectancy': y_pred})

# Print the predictions
print(predictions_df)


# In[35]:


# Data
x = predictions_df['Year']
y = predictions_df['Predicted_Life_Expectancy']

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='g', linestyle='-', label='Predicted Life expectancy')
plt.title('Predicted Life expectancy Rate in Germany ')
plt.xlabel('Year')
plt.ylabel('Predicted Life expectancy Rate (%)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


# In[26]:


predictions_df
import seaborn as sns
sns.pairplot(predictions_df[['Year', 'Predicted_Life_Expectancy']])


# In[42]:


plt.scatter(x, y)
plt.plot(X_fit, Y_fit);


# In[51]:


from sklearn.metrics import mean_squared_error, r2_score


# Evaluate the model
mse = mean_squared_error(Y_fit, y_pred)
rmse = mse ** 0.5
r2 = r2_score(Y_fit, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")


# In[ ]:




