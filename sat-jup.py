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
import sj

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


years = 1000000.
timestep = 1/365.
r_j, v_j = sj.jupiter(years, timestep)
r_s, v_s =sj.saturn(years, timestep)
sj.plotting(years, timestep, number=1000, zoom=10)
print("end of code")
e = time()
print("time taken:",e-s)
