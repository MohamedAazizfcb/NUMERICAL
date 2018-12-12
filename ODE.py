#INPUT : STill WOrking oN It
#OUTPUT : plt.plot(t,y,'r-',t,yasol(t),'b-')
import numpy as np
import matplotlib.pyplot as plt

x0 = 0
y0 = 1
xf = 10
n = 101
delta=(xf-x0)/n
x = np.linspace ( x0 , xf , n )
y = np.zeros ( [ n ] )
y [ 0 ] = y0
for i in range ( 1 , n ) :
   y [i]= delta * (-y [ i-1] +np.sin(x[i-1]))+y[i-1]
for i in range ( n ) :
   print ( x [ i ] , y [ i ] )

plt.plot(x,y,'o')
plt.xlabel (" Value of x " )
plt.ylabel (" Value of y " )
plt.title ("Approximate S ol u ti o n with Forward Euler ’ s Method " )
plt.show ( )



#----------------------------------------------------------------------
#
# heun.py
#
# calculate the curve which is the solution to an ordinary differential
# equation with an initial value using Heun's method
#
=

import math
import numpy as np
import matplotlib.pyplot as plt

# we will use the differential equation y'(t) = y(t).  The analytic solution is y = e^t.

def y1(t,y):
    return y

def asol(t):
    return math.exp(t)

yasol = np.vectorize(asol)

h = 0.5
t0 = 0.0
y0 = 1.0

t = np.arange(0.0, 5.0, h)
y = np.zeros(t.size)
y[0] = y0

for i in range(1, t.size):
    y_intermediate = y[i-1] + h*y1(t[i-1],y[i-1])

    y[i] = y[i-1] + (h/2.0)*(y1(t[i-1],y[i-1]) + y1(t[i],y_intermediate))
    

plt.plot(t,y,'r-',t,yasol(t),'b-')


