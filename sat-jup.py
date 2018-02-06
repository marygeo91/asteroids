#! /bin/env/ python

#initial imports

import sj
from time import time

s = time()

years = 10000.
timestep = 1/365.
r_j, v_j = sj.jupiter(years, timestep)
r_s, v_s =sj.saturn(years, timestep)
sj.plotting(years, timestep, number=100, zoom=10)
print("end of code")
e = time()
print("time taken:",e-s)
