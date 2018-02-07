#! /bin/env/ python

#initial imports

import sj
import numpy as np
from time import time

s = time()

years = 100.
timestep = 1/365.
r_j, v_j = sj.jupiter(years, timestep)
r_s, v_s =sj.saturn(years, timestep)
a=sj.plotting(years, timestep, number=1000, zoom=10)
np.savetxt("jupitersaturnastroidData.txt",a)
print("end of code")
e = time()
print("time taken:",e-s)
