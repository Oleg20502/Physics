#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[135]:


A1 = np.array(
    [5.162, 5.112, 5.051, 5.101, 5.066, 5.130, 5.124, 5.029, 5.110, 5.099, 5.135, 5.106, 5.161, 5.090,
     5.064, 5.112, 5.095, 5.080, 5.036, 5.153, 5.148, 5.108, 5.123, 5.130, 5.118, 5.082, 5.167, 5.041,
     5.158, 5.063, 5.114, 5.116, 5.118, 5.122, 5.157, 5.117, 5.117, 5.160, 5.113, 5.108, 5.141, 5.100,
     5.136, 5.103, 5.106, 5.093, 5.116, 5.105, 5.170, 5.074, 5.110, 5.167, 5.074, 5.096, 5.101, 5.085,
     5.108, 5.079, 5.163, 5.077, 5.108, 5.108, 5.125, 5.121, 5.112, 5.119, 5.108, 5.113, 5.099, 5.120,
     5.093, 5.145, 5.117, 5.091, 5.104, 5.116, 5.092, 5.069, 5.120, 5.167, 5.087, 5.114, 5.085, 5.114,
     5.055, 5.118, 5.136, 5.101, 5.117, 5.126, 5.080, 5.116, 5.036, 5.058, 5.045, 5.118, 5.082, 5.158,
     5.104, 5.132, 5.129, 5.208, 5.105, 5.088, 5.120, 5.116, 5.030, 5.100, 5.120, 5.077, 5.197, 5.104,
     5.068, 5.104, 5.125, 5.128, 5.053, 5.123, 5.138, 5.114, 5.107, 5.149, 5.142, 5.100, 5.124, 5.042,
     5.176, 5.108, 5.099, 5.092, 5.176, 5.154, 5.103, 5.083, 5.023, 5.151, 5.114, 5.121, 5.103, 5.083,
     5.131, 5.046, 5.100, 5.127, 5.062, 5.155, 5.131, 5.119, 5.141, 5.103, 5.114, 5.108, 5.085, 5.120,
     5.099, 5.118, 5.126, 5.099, 5.114, 5.125, 5.123, 5.130, 5.102, 5.113, 5.039, 5.104, 5.088, 5.094,
     5.066, 5.100, 5.126, 5.132, 5.151, 5.109, 5.044, 5.133, 5.088, 5.069, 5.120, 5.127, 5.103, 5.087,
     5.049, 5.117, 5.157, 5.098, 5.150, 5.107, 5.092, 5.082, 5.115, 5.119, 5.061, 5.105, 5.126, 5.141,
     5.078, 5.089, 5.038, 5.087, 5.097, 5.095, 5.199, 5.150, 5.097, 5.087, 5.098, 5.113, 5.130, 5.123,
     5.156, 5.068, 5.152, 5.153, 5.041, 5.155, 5.135, 5.139, 5.108, 5.095, 5.108, 5.112, 5.128, 5.100,
     5.118, 5.098, 5.116, 5.138, 5.118, 5.123, 5.037, 5.108, 5.106, 5.102, 5.136, 5.180, 5.111, 5.270,
     5.084, 5.057, 5.101, 5.082, 5.118, 5.072, 5.149, 5.053, 5.096, 5.171, 5.067, 5.098, 5.138, 5.087,
     5.055, 5.064, 5.109, 5.037, 5.135, 5.122, 5.120, 5.144, 5.118, 5.123, 5.130, 5.110, 5.131, 5.036,
     5.073, 5.116, 5.153, 5.073, 5.106
])


# In[136]:


A2 = np.array(
    [5.162, 5.112, 5.051, 5.101, 5.066, 5.130, 5.124, 5.029, 5.110, 5.099, 5.135, 5.106, 5.161, 5.090,
     5.064, 5.112, 5.095, 5.080, 5.036, 5.153, 5.148, 5.108, 5.123, 5.130, 5.118, 5.082, 5.167, 5.041,
     5.158, 5.063, 5.114, 5.116, 5.118, 5.122, 5.157, 5.117, 5.117, 5.160, 5.113, 5.108, 5.141, 5.100,
     5.136, 5.103, 5.106, 5.093, 5.116, 5.105, 5.170, 5.074, 5.110, 5.167, 5.074, 5.096, 5.101, 5.085,
     5.108, 5.079, 5.163, 5.077, 5.108, 5.108, 5.125, 5.121, 5.112, 5.119, 5.108, 5.113, 5.099, 5.120,
     5.093, 5.145, 5.117, 5.091, 5.104, 5.116, 5.092, 5.069, 5.120, 5.167, 5.087, 5.114, 5.085, 5.114,
     5.055, 5.118, 5.136, 5.101, 5.117, 5.126, 5.080, 5.116, 5.036, 5.058, 5.045, 5.118, 5.082, 5.158,
     5.104, 5.132, 5.129, 5.208, 5.105, 5.088, 5.120, 5.116, 5.030, 5.100, 5.120, 5.077, 5.197, 5.104,
     5.068, 5.104, 5.125, 5.128, 5.053, 5.123, 5.138, 5.114, 5.107, 5.149, 5.142, 5.100, 5.124, 5.042,
     5.176, 5.108, 5.099, 5.092, 5.176, 5.154, 5.103, 5.083, 5.023, 5.151, 5.114, 5.121, 5.103, 5.083,
     5.131, 5.046, 5.100, 5.127, 5.062, 5.155, 5.131, 5.119, 5.141, 5.103, 5.114, 5.108, 5.085, 5.120,
     5.099, 5.118, 5.126, 5.099, 5.114, 5.125, 5.123, 5.130, 5.102, 5.113, 5.039, 5.104, 5.088, 5.094,
     5.066, 5.100, 5.126, 5.132, 5.151, 5.109, 5.044, 5.133, 5.088, 5.069, 5.120, 5.127, 5.103, 5.087,
     5.049, 5.117, 5.157, 5.098, 5.150, 5.107, 5.092, 5.082, 5.115, 5.119, 5.061, 5.105, 5.126, 5.141,
     5.078, 5.089, 5.038, 5.087, 5.097, 5.095, 5.199, 5.150, 5.097, 5.087, 5.098, 5.113, 5.130, 5.123,
     5.156, 5.068, 5.152, 5.153, 5.041, 5.155, 5.135, 5.139, 5.108, 5.095, 5.108, 5.112, 5.128, 5.100,
     5.118, 5.098, 5.116, 5.138, 5.118, 5.123, 5.037, 5.108, 5.106, 5.102, 5.136, 5.180, 5.111, 5.084, 
     5.057, 5.101, 5.082, 5.118, 5.072, 5.149, 5.053, 5.096, 5.171, 5.067, 5.098, 5.138, 5.087, 5.055, 
     5.064, 5.109, 5.037, 5.135, 5.122, 5.120, 5.144, 5.118, 5.123, 5.130, 5.110, 5.131, 5.036, 5.073,
     5.116, 5.153, 5.073, 5.106
])

# здесь нет r = 5.270 кОм. Удалил, чтобы сошлось.


# In[137]:


#R = np.random.normal(5.1, 0.05, 270)
R = np.sort(A2)
print(len(R))
Rm = np.mean(R)
print(Rm)
si_r = round(np.std(R), 2)
print(si_r)


# In[138]:


print(*R)


# In[133]:


Rmin = np.min(R)
Rmax = np.max(R)
X = np.arange(4.95, 5.251, 0.001)
Y = 1/np.sqrt(2*np.pi)/si_r*np.exp(-(X - Rm)**2/2/si_r**2)
plt.subplot()
plt.plot(X, Y)
m1 = 20
dR1 = (Rmax - Rmin)/m1
H1 = plt.hist(R, m1, density=1)
plt.title('Рис. 1. Гистограмма для m = 20 интервалов')
plt.ylabel('w   y')
plt.xlabel('R, кОм')
#сохраняем гистограмму
#plt.savefig('C:/Users/User/Desktop/Олег/Гистограмма для 20 интервалов.png', format='png', dpi=100)
plt.show()
#печатает дульта н, если density=0 и омега, если density=1.
print(np.round(H1[0], 3))


# In[134]:


plt.subplot()
plt.plot(X, Y)
m2 = 10
H2 = plt.hist(R, m2, density=1)
plt.ylabel('w   y')
plt.xlabel('R, кОм')
plt.title('Рис. 2. Гистограмма для m = 10 интервалов')
#сохраняем гистограмму
#plt.savefig('C:/Users/User/Desktop/Олег/Гистограмма для 10 интервалов.png', format='png', dpi=100)  
plt.show()
#печатает дульта н, если density=0 и омега, если density=1.
print(np.round(H2[0], 3))


# In[139]:


Rmin = np.min(R)
Rmax = np.max(R)
bins = [Rmin-0.1, Rm-3*si_r, Rm-2*si_r, Rm-si_r, Rm, Rm+si_r, Rm+2*si_r, Rm+3*si_r, Rmax+0.1]
H3 = plt.hist(R, bins, density=1, stacked = 1)
P1 = sum(H3[0][3:5])*si_r  #вероятность попад. в +- сигма
P2 = sum(H3[0][2:6])*si_r    #вероятность попад. в +- 2 сигма
P3 = sum(H3[0][1:7])*si_r

print(P3)


# In[126]:




