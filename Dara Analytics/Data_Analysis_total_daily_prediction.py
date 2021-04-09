#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


# In[16]:


cases = pd.read_csv('/Users/acer/Downloads/done.csv')[['date','cases']] #to read .csv files ['date', cases] -> to extract data in 'date' and 'cases' rows n columns

cases['date'] = pd.to_datetime(cases['date'], format='%d/%m/%Y') #change the string "date" in csv files to proper format, example "01/01/2020" string to 2020-01-01 -< proper format.

print(cases) #print


# In[17]:


cases['cases'].plot() #to plot the amount of "cases" in csv files


# In[18]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(cases['cases']) #to plot correlation


# In[ ]:


#for total cases prediction


# In[19]:


X = cases['cases'].values    # values in "cases" being put into arrays of data
train = X[0:426]             # data training
real_data = X[426:]          # value of real data
predictions = []            


# In[20]:


real_data.size                # value of total cases (real data)


# In[21]:


from statsmodels.tsa.arima_model import ARIMA


# In[22]:


#p,d,q     p = periods taken for autoregressive model
# d = order of integration, difference
# q = moving average
model_arima = ARIMA(train, order=(7,1,0))         #creating prediction using module
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[23]:


predictions = model_arima_fit.forecast(steps=7)[0]      # prediction data, steps = amount of days for prediction
predictions


# In[24]:


plt.plot(real_data)
plt.plot(predictions,color='red')     # blue line = real data, red line = prediction


# In[ ]:


#for daily cases change prediction


# In[ ]:


daily_change = cases['cases'].diff(periods=1) #daily cases changes periods = 1 means the changes are calculated daily
#integrated of order 1
daily_change = cases_change[1:] #exclude the first day


# In[26]:


print(cases_change)


# In[27]:


plot_acf(cases_change) 


# In[44]:


Y = cases_change.values    # values in "cases" being put into arrays of data
train_new = Y[0:425]             # data training
real_data_2 = Y[424:]          # value of real data
predictions_2 = []


# In[45]:


real_data_2.size


# In[46]:


model_arima = ARIMA(train_new, order=(7,1,0))         #creating prediction using module
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[47]:


predictions_2 = model_arima_fit.forecast(steps=8)[0]      # prediction data, steps = amount of days for prediction
predictions_2


# In[48]:


plt.plot(real_data_2)
plt.plot(predictions_2,color='green')    # blue line = real data, green line = prediction


# In[ ]:





# In[ ]:




