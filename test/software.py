import csv as csv
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
from sklearn import preprocessing
from time import perf_counter, sleep

class Software:

   #class variables
   fileID = 0

   def __init__(self):
       Software.fileID += 1
       self.data = None
       self.target = None
       self.segmented = False

    
   def inputModule(self, csvName): #used to read in training data
        # read CSV Data into a numpy array
        
        # data => accel_x, accel_y, accel_z
        start = perf_counter()
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
        #sleep(1)
        end = perf_counter()
        
        #print(self.target[1:10])
        print(self.data.shape[0])
        self.benchmark(start, end, '*** Read From CSV ***')
        self.preprocessingModule()
        
   def segmentationModule(self):
       print('not done')
       
   def preprocessingModule(self):
       if self.data is not None and self.segmented:
           #normalize signal data
           self.norm_data = preprocessing.normalize(self.data)
           print(self.norm_data[:10])
       else:
           print('data not ready for preprocessing')
                   
   def benchmark(self, start, end, message = False): #used to determine performance of algorithm
       if message:
           print(message + '\nTime Elapsed: '+ " %.9f seconds" % (end-start) + '\n')
       else:
           print('\nTime Elapsed: '+ " %.3f seconds" % (end-start) + '\n')
        