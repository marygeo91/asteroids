#! /bin/env/ python

#initial imports
import numpy as np
from numpy.random import random
from numpy.random import randint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from time import time
from matplotlib import animation

# functions
s = time()
# Jupiter
a_jup = 5.204            # Semi-major-axis (units of AU)
e_jup = 0.0489           # Eccentricity 

# Saturn
a_sat = 9.582          # Semi-major-axis (units of AU)
e_sat = 0.0565           # Eccentricity

G = 4*np.pi**2           # Gravitational constant (units of M_sun, AU and year)
M_sun = 1.               # Solar mass 
M_j = 1/1047.*M_sun      # Fraction of M_sun
M_s = 568.34*10**24/(1.989*10**30)*M_sun      # Fraction of M_sun 568.34
h = 0.01

def jupiter(time,h):
    '''This function applies the Euler-Cromer (or Symplectic Integrator) 
    method for solving differntial equations'''
    # position and velocity arrays 
    n = int(time/h)
    r_jup = np.zeros((3,n))
    v_jup = np.zeros((3,n))
    radius = np.zeros(n)
    accel_sun = np.zeros(n)

    # initial conditions

    r_jup[0,0] = 0.0
    r_jup[1,0] = a_jup*(1-e_jup)
    r_jup[2,0] = 0.0
    v_jup[0,0] = -np.sqrt((G*M_sun/a_jup) * (1+e_jup)/(1-e_jup))
    v_jup[1,0] = 0.0
    v_jup[2,0] = 0.0

    for t in range(n-1):
        radius[t] = np.sqrt(np.sum(r_jup[:,t]**2))
        accel_sun[t]= -G*M_sun/(radius[t]**3)
        v_jup[:,t+1] =  v_jup[:,t] + h*accel_sun[t]*r_jup[:,t]
        r_jup[:,t+1] =  r_jup[:,t] + h*v_jup[:,t+1] 
                
    return r_jup,v_jup

def saturn(time,h):
    '''This function applies the Euler-Cromer (or Symplectic Integrator) 
    method for solving differntial equations'''

    # position and velocity arrays
    n = int(time/h)  
    r_sat = np.zeros((3,n))
    v_sat = np.zeros((3,n))
    radius = np.zeros(n)
    accel_sun = np.zeros(n)
    
    # initial conditions
    r_sat[0,0] = 0.0
    r_sat[1,0] = a_sat*(1-e_sat)
    r_sat[2,0] = 0.0
    v_sat[0,0] = -np.sqrt((G*M_sun/a_sat) * (1+e_sat)/(1-e_sat))
    v_sat[1,0] = 0.0
    v_sat[2,0] = 0.0
    
    for t in range(n-1):
        radius[t] = np.sqrt(np.sum(r_sat[:,t]**2))
        accel_sun[t]= -G*M_sun/(radius[t]**3)
        v_sat[:,t+1] =  v_sat[:,t] + h*accel_sun[t]*r_sat[:,t]
        r_sat[:,t+1] =  r_sat[:,t] + h*v_sat[:,t+1] 
        
    return r_sat,v_sat

def symplectic_astroid(time,h):
     '''This function applies the Euler-Cromer (or Symplectic Integrator) 
     method for solving differntial equations'''
     n = int(time/h)  
     r = np.zeros((3,n))
     v = np.zeros((3,n))
     radius = np.zeros(n)
     radius_j = np.zeros(n)
     radius_s = np.zeros(n)
     accel_sun = np.zeros(n)
     accel_j = np.zeros(n)
     accel_s = np.zeros(n)
     radius0 = np.random.uniform(2.0,3.5)
     theta0 = np.random.uniform(0.0,2*np.pi)
     
     r[0,0] = radius0*np.cos(theta0)
     r[1,0] = radius0*np.sin(theta0)
     r[2,0] = (1/2000.)*radius0*np.sin(theta0)
     v[0,0] = -np.sqrt((G/radius0)*(1+e_jup)/(1-e_jup))*np.sin(theta0) 
     # velocity more influenced by Jupiter
     v[1,0] = np.random.uniform(0.0,2*np.pi)
     v[2,0] = 0.000001
     
     for t in range(0,n-1):
         radius[t] = np.sqrt( np.sum(r[:,t]**2))
         
         if (radius[t] < 1 or radius[t] > 5) : 
             #r = np.delete(r, (0,1,2), axis=0)
             #v = np.delete(v, (0,1,2), axis=0)
             break
         
         radius_j[t] = np.sqrt(np.sum((r[:,t]-r_j[:,t])**2))
         accel_j[t] = -G*M_j/(radius_j[t]**3) 
         radius_s[t] = np.sqrt(np.sum((r[:,t]-r_s[:,t])**2))
         accel_s[t] = -G*M_s/(radius_s[t]**3) 
         accel_sun[t]= -G*M_sun/(radius[t]**3)        
         v[:,t+1] =  v[:,t] + h*(accel_j[t]*(r[:,t]-r_j[:,t])+accel_sun[t]*r[:,t])
         r[:,t+1] =  r[:,t] + h*v[:,t+1]
         
         return r, v, radius


def plotting(time,h,number=1,zoom=6.0):
    fig = plt.figure(figsize=(20,10)) 
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*r_j)
    ax.plot(*r_s)
    
    for i in range(number):
        r_as, v_as, rad_as = symplectic_astroid(time,h)
        
        if (r_as.any() == True):
            ax.plot(*r_as)
        
    x = r_as[0,:]
    y = r_as[1,:]
    z = r_as[2,:]
    radius = rad_as
    np.savetxt('jupiterandsaturnasteroidData.txt', np.c_[x, y, z, radius])
    ax.set_xlabel("x position", fontsize=16)
    ax.set_ylabel("y position",fontsize=16)
    ax.legend(loc = 'best', fontsize = 15)

    ax.set_title("time= " + str(time) + ", h= " +str(h))
    ax.set_aspect('equal','datalim')
    ax.set_xlim(-zoom,zoom)
    ax.set_ylim(-zoom,zoom)
    ax.set_zlim(-0.1,0.1)
    plt.show()
    
years = 1000
timestep = 1/365.
r_j, v_j = jupiter(years, timestep)
r_s, v_s = saturn(years, timestep)
plotting(years, timestep, number=1000, zoom=10)
print("end of code")
e = time()
print("time taken:",e-s)
