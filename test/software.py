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

   def __init__(self, debug = False):
       Software.fileID += 1
       
       self.debug = debug
       self.rawData = None
       self.data = None
       self.target = None
       
      
    
   def inputModule(self, csvName): #used to read in training data
        # read CSV Data into a numpy array
        # consider looping once using default CSV library into numpy array
        
        start = perf_counter()
        self.rawData = np.genfromtxt(csvName, delimiter=',', usecols= range(1,5), dtype= int)
        
        # columns:   0   |    1   |   2    |    3
        # data => accel_x, accel_y, accel_z, identifier
        
        #               identifers
        #    --- Labels are codified by numbers
        #    --- 1: Working at Computer
        #    --- 2: Standing Up, Walking and Going up\down stairs
        #    --- 3: Standing
        #    --- 4: Walking
        #    --- 5: Going Up\Down Stairs
        #    --- 6: Walking and Talking with Someone
        #    --- 7: Talking while Standing
        
        #self.target = np.genfromtxt(csvName, delimiter=',', usecols= (4), dtype= int)
        #sleep(1)
#        iris = load_iris()
#        X = iris.data
#        print(X)
        
        end = perf_counter()
        
        #print(self.target[1:10])
        #print(self.data.shape[0])
        
        self.benchmark(start, end, '*** Read From CSV ***')
        
        start = perf_counter()
        self.segmentationModule(self.rawData, 5)
        end = perf_counter()
        
        self.benchmark(start, end, '*** Segmentation Module ***')
        #seg = self.segment_signal(self.rawData, 5) #proves that output signal is a 3 dimensional numpy array
        #print(seg[:10])
        
        start = perf_counter()
        self.preprocessingModule()
        end = perf_counter()
        
        self.benchmark(start, end, '*** Preprocessing Module ***')
        
   def segment_signal(self, data, window_size):
       
       N = data.shape[0]
       dim = data.shape[1]
       K = int(N/window_size)
       
       segments = np.empty((K, window_size, dim))
       
       for i in range(K):
           segment = data[i*window_size:i*window_size+window_size,:]
           segments[i] = np.vstack(segment)
       return segments
       
        
   def segmentationModule(self, data, frame_size): 
       #frame_size refers to the number of samples per frame, no overlap
       #frame_size corresponds to the period given for example a sampling rate of 52Hz for particular data set
       
       if data is not None:
           
           #obtain number of rows of data / data samples
           N = data.shape[0]
           
           #obtain number of features
           dim = data.shape[1] - 1 #last column is identifier
           
           #indexes
           i = 0
           K = 0
           
           #Count number of valid segments
           while((i + frame_size) < N):
               
               #if samples in frame consists of the same classifier
               if data[i][3] == data[i + frame_size - 1][3]:
                   #increment frame number
                   K += 1
                   
                   #move to next frame
                   i += frame_size           
                   
               else:
                   i += 1
           
           #print("Number of valid frames: " + str(K))
           
           #create empty numpy array
           segments = np.empty((K, frame_size, dim))
           self.norm_data = np.empty((K, frame_size, dim)) #for preprocessing later
                     
           #store as instance variable
           self.numberOfSegments = K
           
           #indexes
           i = 0
           j = 0
           
           #target list, to be converted into a numpy array
           target = []
           
           #print("#### \n")
           #iterate across each row 
           while((i + frame_size) < N):
               
               #if samples in frame consists of the same classifier
               if data[i][3] == data[i + frame_size - 1][3]:
                   
                   #sample frame aka segment consists of only up to the 3rd column
                   segment = np.vstack(data[i:i + frame_size, :3])
                   segments[j] = segment
                   
                   #update list of identifiers corresponding to particular sample frame
                   target.append(data[i][3])
                   
                   #move to next frame
                   i += frame_size
                   
                   #move counter for 
                   j += 1
                   
               else:
                   i += 1
           
           
           #determine number of valid frames
           valid_frames = len(target)
           actual_frames = segments.shape[0]
           
           
           
           #convert target to numpy array
           self.target = np.asarray(target)
           self.data = segments
           
           if self.debug:
               print("Calculated Target frames: " + str(valid_frames) + " Data frames: " + str(actual_frames) + "\n")
               print("Target Data:")
               print(target[:10])
               print("Segmented Data:")
               print(segments[-10:])
           
            
       
   def preprocessingModule(self): #preprocessing can be done inline with the segmentation  
       if self.data is not None and self.numberOfSegments is not None:
           
           #normalize signal data
           #self.norm_data = preprocessing.normalize(self.data[0])
           
           for i in range(self.numberOfSegments):
               self.norm_data[i] = preprocessing.normalize(self.data[i])
               
           if self.debug:       
               print("### normalised data set ###")    
               print(self.norm_data[:10])
           
       else:
           print('data not ready for preprocessing')
           
   def featureExtractionModule(self, data):
       # 12 features will be selected from the accelerometer readings
       # to use the data for classification, we need to convert the data set into a 2D NumPy array 
       
       #Averages
       #1. Average X    2. Average Y   3. Average Z
       
       #Standard Deviations
       #4. Standard deviation X 5. Standard deviation Y 6. Standard deviation Z
       
       #Covariance
       #7. Covariance X, Y    8. Covariance Y, Z    9. Covariance Z, X
       
       #Correlation
       #10. Correlation X, Y   11. Correlation Y, Z    12. Correlation Z, X
       for i in range(self.numberOfSegments):
           print("placeholder")
       
       
       
       
   def benchmark(self, start, end, message = False): #used to determine performance of algorithm
       if message and self.debug:
           print(message + '\nTime Elapsed: '+ " %.9f seconds" % (end-start) + '\n')
       else:
           print('\nTime Elapsed: '+ " %.3f seconds" % (end-start) + '\n')
        