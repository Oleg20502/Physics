#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


t = np.arange(-2.5, 2, 0.01)
X = t**3+2*t**2 + t
Y = -t**3 + 3*t -2
plt.plot(X, Y)
plt.show()


# In[16]:


Ax = 2
Ay = 2
wx = 2
wy = 3
fix = 0
fiy = 0

t = np.arange(-3.14, 3.15, 0.01)
X = Ax * np.sin(wx*t + fix)
Y = Ay * np.sin(wy*t + fiy)
plt.plot(X, Y)
plt.show()


# In[ ]:




