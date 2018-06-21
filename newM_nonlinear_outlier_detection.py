import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

num=800
x = np.linspace(0,1,num)
y =  np.exp(x)
y = y / np.max(y)
yo = y.copy()

# two positive outliers
y[200:215] = y[200:215] + np.linspace(1,15, num=15) / 1000
y[215:230] = y[215:230] + 0.95 * np.linspace(15,1, num=15) / 1000
y[400:420] = y[400:420] + np.linspace(1,20,num=20) / 1000
y[420:440] = y[420:440] + 0.95 * np.linspace(20,1,num=20) / 1000


# two negative outliers
y[600:615] = y[600:615] - np.linspace(1,15, num=15) / 1000
y[615:630] = y[615:630] - 0.95 * np.linspace(15,1, num=15) / 1000
'''
y[800:820] = y[800:820] - np.linspace(1,20,num=20) / 100
y[820:840] = y[820:840] - np.linspace(20,1,num=20) / 100
'''

noise = 0.00001* np.random.randn(num)
y_n = y + noise     # with outliers 
yo_n = yo+ noise    # without outliers


plt.figure(1)
plt.subplot(121)
plt.plot(x,y_n,'*')

plt.subplot(122)
plt.plot(x,yo_n,'*')
plt.show()

'''
# difference of y
d_x = 1 / num
d_y=np.zeros(num-1)
d_yo=np.zeros(num-1)
for i in range(num-1):
    d_y[i] = (y_n[i+1] - y_n[i]) / d_x
    d_yo[i] = (yo_n[i+1] - yo_n[i]) / d_x

# upper and lower thresholds of d_y based on mean and std 
y_up = np.mean(d_y) + 3 * np.std(d_y) * np.ones(num-1)
y_lo = np.mean(d_y) - 3 * np.std(d_y) * np.ones(num-1)

yo_up = np.mean(d_yo) + 3 * np.std(d_yo) * np.ones(num-1)
yo_lo = np.mean(d_yo) - 3 * np.std(d_yo) * np.ones(num-1)


# upper and lower threshold d_y based on median and robsd
robsd = np.median(np.abs(d_y - np.median(d_y)))
y_mup = np.median(d_y) + 3 * robsd * np.ones(num-1)
y_mlo = np.median(d_y) - 3 * robsd * np.ones(num-1)

#scr = np.absolute(d_y - np.median(d_y)) / robsd
#nscro = (scr - np.min(scr)) / (np.max(scr) - np.min(scr))

robsdo = np.median(np.abs(d_yo - np.median(d_yo)))
yo_mup = np.median(d_yo) + 2.5 * robsdo * np.ones(num-1)
yo_mlo = np.median(d_yo) - 2.5 * robsdo * np.ones(num-1)

#scro = np.absolute(d_yo - np.median(d_yo)) / robsdo
#nscro = (scro - np.min(scr)) / (np.max(scr) - np.min(scr))

plt.figure(2)
plt.subplot(121)
plt.plot(x[:num-1],d_y,'*b',
         x[:num-1],y_mup, 'r', x[:num-1], y_mlo,'r',
         x[:num-1],y_up,'k',x[:num-1],y_lo,'k')

plt.subplot(122)
plt.plot(x[:num-1],d_yo,'*b')
plt.show()
'''

def nHI(t,HI,L,a)
# t is time
# HI is the difference of the original health index;
# L is the length of a sub-region
# 1-a is cross ratio
    if len(t) != len(HI):
        print('The lengths of t and HI should be the same')
    
    num = len(HI)
    cHI = HI.copy()
    d_t = (t[num-1] - t[0]) / num
    dHI = [(HI[i+1]- HI[i]) / d_t for i in range(num-1)]

    remainder = (num - 1) % L
    if  remainder == 0:
        sr_n = (num - 1) // L
        sr = np.zeros((sr_n,L))
        for i in range(sr_n):
            sr[i,:] = dHI[i*L:(i+1)*L]
            
    else:
        sr_n = (num - 1) // L + 1
    
    

    

























