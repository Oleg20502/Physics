#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[9]:


def k_coef_count(x, y, dig_r = 3):
    k = np.round(np.mean(y*x)/np.mean(x**2), dig_r)
    return k

def si_k_count(x, y, dig_r = 4):
    n = len(x)
    si = np.round(np.sqrt(((np.mean(x**2) * np.mean(y**2) - np.mean(x*y)**2) / n / np.mean(x**2)**2)),
                  dig_r)
    return si

def si_rand_count(x, dig_r = 3):
    x0 = np.mean(x)
    n = len(x)
    si = np.round(np.sqrt(1/(n-1) * np.sum((x - x0)**2)), dig_r)
    return si

def si_multi(*a, dig_r=3):
    d = 0
    for i in range(1, len(a), 3):
        d += (a[i+2] * a[i] / a[i+1])**2
    si = np.round(a[0] * np.sqrt(d), dig_r)
    return si


# In[79]:


Fr = np.array([1.03, 10.85, 39.96, 79.45, 102.17])   #кГц
si_Fr = 0.01
Div = np.array([5, 4.6, 5, 2.6, 5])  #деления
si_Div = 0.2
T_div = np.array([200, 20, 5, 5, 2])   #мкс

T = np.round(Div * T_div , 3)
si_T = np.round(T_div * si_Div , 3)
print(T, si_T)

F = np.round(1/T*10**3, 2)
si_F = np.round(F/T * si_T, 2)
print(F, si_F)

dF = np.abs(Fr - F)
print(dF)


# In[84]:


divv_min = 3
divv_max = 2
V_div_min = 0.02
V_div_max = 5

Vmin = np.round(divv_min * V_div_min, 3)
si_Vmin = np.round(si_Div * V_div_min, 3)
sip_Vmin = np.round(si_Vmin/Vmin, 4)
Vmax = np.round(divv_max * V_div_max, 2)
si_Vmax = np.round(si_Div * V_div_max, 3)
sip_Vmax = np.round(si_Vmax/Vmax, 4)
print(Vmin, si_Vmin, sip_Vmin, Vmax, si_Vmax, sip_Vmax)

n = np.round(20 * np.log10(Vmax / Vmin), 0)
si_n = np.round(20/np.log(10) * np.sqrt((si_Vmin/Vmin)**2 + (si_Vmax/Vmax)**2), 1)
print(n, si_n)


# In[85]:


f1 = np.array([1, 3, 5, 7, 10, 
              100, 1000, 10**4, 10**5, 10**6, 10**7, 
              2*10**7, 2.5*10**7, 2.7*10**7, 3*10**7])
si_f1 = 0.1
flg1 = np.round(np.log10(f1), 3)
print(flg1)

V0 = np.array([4]*15)

#Uac = np.array([1.2, 2.4, 3.2, 3.6, 3.6, 4, 4, 4, 4, 4, 4, 3.6, 3.2, 2.8, 2.4])
Uac = np.array([1.2, 2.4, 2.8, 3.2, 3.6, 4, 4, 4, 4, 4, 4, 3.6, 3.2, 2.8, 2.4])
Udc = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3.6, 3.2, 2.8, 2.4])
Kac = Uac/V0
Kdc = Udc/V0
print(Kac,'\n', Kdc)


# In[89]:


plt.plot(flg1, Kac, 'ob')
plt.plot(flg1, Kdc, 'or')
plt.grid()
plt.show()


# In[55]:


f2 = np.array([142, 565, 1070, 1525, 2633, 3363, 3840, 4619, 5269])
flg2 = np.round(np.log10(f2), 3)
print(flg2)


Y = np.array([0.4, 1.2, 2.4, 3.2, 4, 3.2, 2.4, 1.2, 0.4])
A = np.array([4.4, 4.4, 4.4, 4.4, 4, 4, 4, 4, 4])

dfi1 = np.round(np.arcsin(Y/A), 2)
dfi2 = np.zeros(len(dfi1))
dfi2[:5] = dfi1[:5]
dfi2[5:] = 3.14 - dfi1[5:]
print(dfi2)


# In[70]:


plt.plot(flg2, dfi2, 'og')
plt.plot(flg2, dfi2)
plt.grid()
plt.show()


# In[ ]:




