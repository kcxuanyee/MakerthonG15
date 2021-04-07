#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


# In[10]:


def parser(x):
    return datetime.strptime(x, '%d/%m/%Y')

cases = pd.read_csv('/Users/acer/Downloads/covid.csv', index_col=0, parse_dates=[0], date_parser=parser) #change the file location
cases = cases.diff(periods=1)
cases_diff = cases[1:]
print(cases_diff)


# In[11]:


cases_diff.plot()


# In[15]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(cases_diff)


# In[109]:


cases_change = cases_diff.diff(periods=1)
#integrated of order 1
cases_change = cases_change[1:]


# In[110]:


print(cases_change)


# In[111]:


plot_acf(cases_change)


# In[112]:


cases_change.plot()


# In[113]:


cases_change.size


# In[114]:


X = cases_change.values
train = X[0:300]   
test = X[300:]      


# In[115]:


predictions = []
test.size


# In[116]:


from statsmodels.tsa.arima_model import ARIMA


# In[117]:


#p,d,q     p = periods taken for autoregressive model
# d = order of integration, difference
# q = moving average
model_arima = ARIMA(train, order=(2,2,0))
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[118]:


predictions = model_arima_fit.forecast(steps=131)[0]
predictions


# In[119]:


plt.plot(test)
plt.plot(predictions,color='red')


# In[69]:


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




