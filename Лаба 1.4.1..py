#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


Data = [152.13, 151.48, 153.22, 152.92, 136.19, 102.43, 89.36, 82.50, 78.15, 76.99, 76.23, 76.39,
        77.51, 78.49, 76.39, 76.99]


# In[4]:


T_a_check = Data[:4]
T_a1 = T_a_check[:2]
T_a2 = T_a_check[2:4]
print(T_a1)
t1 = np.sum(T_a1)/2
print(t1)
t2 = np.sum(T_a2)/2
print(t2)
dt = round(t2-t1, 3)
print(dt)
T1 = np.round(t1/100, 3)
print(T1)
T2 = np.round(t2/100, 3)
print(T2)


# In[5]:


si_t = 0.05
si_a = 0.01
T_l = np.array(Data[4:len(Data)-2])/50
print(T_l)
a = np.arange(5, 41+4, 4)*0.01


# In[6]:


X = a**2
Y = T_l**2 * a
x = X[3:]
y = Y[3:]
si_y = np.round(np.max(y)*np.sqrt(4*(si_t/np.max(T_l))**2 + (si_a/np.max(a))**2), 3)
si_x = np.round(np.max(x)*si_a/np.max(a), 3)
print(si_y, si_x)
plt.errorbar(x, y, xerr=si_x, yerr=si_y)
#plt.plot(x, y)
p, v = np.polyfit(x, y, deg=1, cov=True)
p_f = np.poly1d(p)
#plt.plot(x[:], p_f(x))
plt.title('$T^2a$, $м\cdotс^2$        Рис.1 Зависимость $T^2a$ от $a^2$                 ')
#plt.ylabel('$T^2a$')
plt.xlabel('$a^2$, $м^2$')
plt.grid()
#plt.savefig('C:/Олег/График колебаний.png', format='png', dpi=300)
plt.show()
print(X)
print(Y)
print(np.max(Y))


# In[7]:


p, v = np.polyfit(x, y, deg=1, cov=True)
p_f = np.poly1d(p)
plt.plot(x[:], p_f(x))
plt.grid()
plt.show()
print(p, v)


# In[74]:


plt.errorbar(x, p_f(x), xerr=si_x, yerr=si_y)
plt.show()


# In[85]:


si_b_ac = np.round(np.sqrt(((np.mean(y**2)-np.mean(y)**2)/(np.mean(x**2)-np.mean(x)**2) - p[1]**2)/len(x)), 3)
si_k_ac = np.round(si_b_ac*np.sqrt(np.mean(x**2)-np.mean(x)**2), 3)
print(si_b_ac, si_k_ac)


# In[91]:


print(np.mean(y**2)-np.mean(y)**2)
print(np.mean(y)**2)
print(np.mean(y**2))
print(np.mean(x**2)-np.mean(x)**2)
print(np.mean(x)**2)
print(np.mean(x**2))


# In[98]:


get_ipython().run_line_magic('pinfo', 'np.polyfit')


# In[ ]:




