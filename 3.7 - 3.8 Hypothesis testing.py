#!/usr/bin/env python
# coding: utf-8

# # Hypothesis testing

# In[2]:


import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# import microsoft.csv, and add a new feature - logreturn
ms = pd.read_csv('Data_CSV/microsoft.csv')
ms.set_index('Date',inplace = True) # sets the DataFrame's index to the 'Date' column.
ms['logReturn'] = np.log(ms['Close'].shift(-1)) - np.log(ms['Close'])


# In[18]:


# Log return goes up and down during the period
plt.title("Daily Return of Microsoft from 2014 to 2017", size = 20)
ms['logReturn'].plot(figsize=(20, 8))
plt.axhline(0, color='red')
plt.xlabel("Time", size = 10)
plt.ylabel("Daily Return", size = 10)
plt.show()


# In[19]:


plt.title("Close Price of Microsoft from 2014 to 2017", size = 10)
plt.xlabel("Time", size = 10)
plt.ylabel("US $", size = 10)
plt.plot(ms.loc[:,'Close'])


# In[21]:


plt.title("Histogram of Daily Return of Miscorsoft from 2014 to 2017", size = 10)
ms.loc[:,'logReturn'].dropna().hist(bins = 100,figsize=(10, 4))


# # Standardization

# In[9]:


xbar = ms['logReturn'].mean() # mean of daily log return of Microsoft
s = ms['logReturn'].std(ddof=1)
n = ms['logReturn'].shape[0]
zhat = (xbar-0)/(s/(n**0.5))
print(xbar)
print(zhat)


# In[17]:


alpha = 0.05
zleft = norm.ppf(alpha/2,0,1)
zright = -zleft
print('zleft = ',zleft,', zright =',zright)
print("Rejection region: zhat < {:.4f} or zhat > {:.4f}".format(zleft, zright))
print('At the significance level of ', alpha)
print('Shall we reject?:', zhat<zleft or zhat>zright )


# # Set Decision Criteria of One Tail Test

# In[24]:


alpha = 0.05
zright = norm.ppf(1-alpha,0,1)
print("zriht:{:.4f}, zhat:{:.4f}".format(zright,zhat))
print('At the significance level of ', alpha)
print('Shall we reject?:', zhat>zright )


# # Calculate p value for Two Tails Test in python

# In[13]:


# Null hypothesis mu = 0
mu = 0

# Test statistic (z-score)
zhat = 1.6141477140003675

alpha = 0.05
p = 2 *(1 - norm.cdf(np.abs(zhat), 0, 1))
print('At the significance level of ', alpha,', p value =', p)
print('Shall we reject?:', p<alpha )


# ## Steps involved in testing a claim by hypothesis testing

# ### Step 1: Set hypothesis

# $H_0 : \mu = 0$ 
# $H_a : \mu \neq 0$
# 
# H0 means the average stock return is 0
# H1 means the average stock return is not equal to 0

# ### Step 2: Calculate test statistic

# In[8]:


sample_mean = ms['logReturn'].mean()
sample_std = ms['logReturn'].std(ddof=1)
n = ms['logReturn'].shape[0]

# if sample size n is large enough, we can use z-distribution, instead of t-distribtuion
# mu = 0 under the null hypothesis
zhat = (sample_mean - 0)/(sample_std/n**0.5)
print(zhat)


# ### Step 3: Set desicion criteria

# In[9]:


# confidence level
alpha = 0.05

zleft = norm.ppf(alpha/2, 0, 1)
zright = -zleft  # z-distribution is symmetric 
print(zleft, zright)


# ### Step 4:  Make decision - shall we reject H0?

# In[10]:


print('At significant level of {}, shall we reject: {}'.format(alpha, zhat>zright or zhat<zleft))


# ## Try one tail test  

# $H_0 : \mu \leq 0$ 
# $H_a : \mu > 0$

# In[11]:


# step 2
sample_mean = ms['logReturn'].mean()
sample_std = ms['logReturn'].std(ddof=1)
n = ms['logReturn'].shape[0]

# if sample size n is large enough, we can use z-distribution, instead of t-distribtuion
# mu = 0 under the null hypothesis
zhat = None
print(zhat)


# ** Expected output: ** 1.6141477140003675

# In[12]:


# step 3
alpha = 0.05

zright = norm.ppf(1-alpha, 0, 1)
print(zright)


# ** Expected output: ** 1.64485362695

# In[13]:


# step 4
print('At significant level of {}, shall we reject: {}'.format(alpha, zhat>zright))


# ** Expected output: ** At significant level of 0.05, shall we reject: False

# # An alternative method: p-value

# In[14]:


# step 3 (p-value)
p = 1 - norm.cdf(zhat, 0, 1)
print(p)


# In[15]:


# step 4
print('At significant level of {}, shall we reject: {}'.format(alpha, p < alpha))


# In[ ]:


import scipy.stats as st

# Null hypothesis mu = 0
mu = 0

# Test statistic (z-score)
z = 2.5

# Two-tailed test (Ha: mu != 0)
p_value_two_tailed = 2 * (1 - st.norm.cdf(abs(z), 0, 1))
print("Two-tailed p-value:", p_value_two_tailed)

# Upper-tailed test (Ha: mu > 0)
p_value_upper_tailed = 1 - st.norm.cdf(z, 0, 1)
print("Upper-tailed p-value:", p_value_upper_tailed)

# Lower-tailed test (Ha: mu < 0)
p_value_lower_tailed = st.norm.cdf(z, 0, 1)
print("Lower-tailed p-value:", p_value_lower_tailed)

