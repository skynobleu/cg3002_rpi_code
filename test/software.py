import csv as csv
import numpy as np
import pandas as pd
import scipy as sp

import matplotlib
from sklearn import preprocessing

class Software:

   #class variables
   fileID = 0

   def __init__(self):
      Software.fileID += 1
      self.dataset = None


   def inputModule(self, csvName):


        # read CSV Data into a numpy array

        # data => accel_x, accel_y, accel_z
       self.data = np.genfromtxt(csvName, delimiter=',', usecols= range(1,4), dtype= int)


        #               target
        #    --- Labels are codified by numbers
        #    --- 1: Working at Computer
        #    --- 2: Standing Up, Walking and Going up\down stairs
        #    --- 3: Standing
        #    --- 4: Walking
        #    --- 5: Going Up\Down Stairs
        #    --- 6: Walking and Talking with Someone
        #    --- 7: Talking while Standing

       self.target = np.genfromtxt(csvName, delimiter=',', usecols= (4), dtype= int)


       print(self.target[1:10])
