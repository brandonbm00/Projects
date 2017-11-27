import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from sys import argv
import math as m
import scipy.stats as sci

np.array([1,2,3])

x = pm.Normal('what', 0, 1)

#before = pm.Poisson('before', 1)
#after = pm.Poisson('after', 1.5)



#before_data = [before.random() for i in np.linspace(0,50,1)]
#after_data = [after.random() for i in np.linspace(51,100,1)]
