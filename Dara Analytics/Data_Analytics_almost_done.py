#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


# In[44]:


def parser(x):
    return datetime.strptime(x, '%d/%m/%Y')

cases = pd.read_csv('/Users/acer/Downloads/done.csv', index_col=0, parse_dates=[0], date_parser=parser) #change the file location
print(cases)


# In[45]:


cases.plot()


# In[46]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(cases)


# In[47]:


cases_change = cases.diff(periods=1)
#integrated of order 1
cases_change = cases_change[1:]


# In[94]:


print(cases_change)


# In[95]:


plot_acf(cases_change)


# In[96]:


cases_change.plot()


# In[180]:


X = cases.values
train = X[0:83]   
test = X[82:]    
predictions = []


# In[181]:


test.size


# In[182]:


from statsmodels.tsa.arima_model import ARIMA


# In[196]:


#p,d,q     p = periods taken for autoregressive model
# d = order of integration, difference
# q = moving average
model_arima = ARIMA(train, order=(7,1,0))
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[197]:


predictions = model_arima_fit.forecast(steps=8)[0]
predictions


# In[198]:


plt.plot(test)
plt.plot(predictions,color='red')


# In[81]:


import itertools
p=d=q=range(0,5)
pdq = list(itertools.product(p,d,q))
pdq


# In[65]:


import warnings
warnings.filterwarnings('ignore')
for param in pdq:
    try:
        model_arima = ARIMA(train, order=param)
        model_arima_fit = model_arima.fit()
        print(param,model_arima_fit.aic)
    except:
        continue


# In[ ]:





# In[ ]:




