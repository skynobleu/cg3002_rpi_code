import csv as csv
import numpy as np
import pandas as pd
import scipy as sp
from datetime import datetime
from sklearn.preprocessing import Imputer, normalize
from time import perf_counter, sleep
from sklearn.cross_validation import train_test_split, KFold #for splitting training and test data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


#https://datascience.stackexchange.com/questions/11928/valueerror-input-contains-nan-infinity-or-a-value-too-large-for-dtypefloat32

class Software:
    
    #class variables
    fileID = 0
    
    def __init__(self, debug = False, outputfile = False):
       Software.fileID += 1
       
       self.debug = debug
       self.rawData = None
       self.segmentedData = None
       self.target = None
       self.extractedData = None
       self.outputfile = outputfile
    
    
    def inputModule(self, csvName): #used to read in training data
        # read CSV Data into a numpy array
        # consider looping once using default CSV library into numpy array
        
        start = perf_counter()
        self.rawData = np.genfromtxt(csvName, delimiter=',', usecols= range(0,4), dtype="Float64")

        print("*** CHECK FOR INVALID VALUES READ FROM CSV ***")
        print(self.rawData.shape)
        index = 0
        for i in self.rawData:
            #print(i)
            for j in i:
                if not np.isfinite(j) or np.isnan(j):
                    print("Invalid Value At: ")
                    print(index, i)
                    self.rawData[index] = np.nan_to_num(i)
                    print("Replaced With: ")
                    print(index, self.rawData[index])
                    print("\n")
            index += 1


        end = perf_counter()
 
        self.benchmark(start, end, '*** Read From CSV ***')
        
        start = perf_counter()
        self.segmentationModule(self.rawData, 150, True)
        end = perf_counter()
        
        self.benchmark(start, end, '*** Segmentation Module ***')
        # seg = self.segment_signal(self.rawData, 5) #proves that output signal is a 3 dimensional numpy array
        #print(seg[:10])
        
        # start = perf_counter()
        # self.preprocessingModule(self.segmentedData)
        # end = perf_counter()
        #
        # self.benchmark(start, end, '*** Preprocessing Module ***')
        
        start = perf_counter()
        self.featureExtractionModule(self.segmentedData)
        #self.featureExtractionModule(self.normData)
        end = perf_counter()
        
        self.benchmark(start, end, '*** Feature Extraction Module ***')
        
        # print("size of input: " + str(self.extractedData.shape[0]) + " size of target: "+ str(self.target.shape[0]) + "\n")

        #Determine if ther are any invalid values

        print("***  Rectify Invalid Values in extractedData ***")
        index = 0
        for i in self.extractedData:
            #print(i)
            for j in i:
                if not np.isfinite(j) or np.isnan(j):
                    print("Invalid Value In extractedData At: ")
                    print(index, i)
                    self.extractedData[index] = np.nan_to_num(i)
                    print("Replaced With: ")
                    print(index, self.extractedData[index])
                    print("\n")
            index += 1
        
        X = self.extractedData
        y = self.target

        # Split dataset into test data and training data
        start = perf_counter()
        X_train, X_test, y_train, y_test = train_test_split(self.extractedData, self.target, random_state=5)
        end = perf_counter()
        self.benchmark(start, end, '*** Splitting datasets into test and training ***')

        f = open(self.outputfile, 'w')
        # dataset information
        print('Learning Model Ran At: ' , str(datetime.now()), file=f)
        print("X_train shape: {}".format(X_train.shape), file=f)
        print("y_train shape: {}".format(y_train.shape), file=f)
        
        print("Size of training set: {} size of test set: {}".format(X_train.shape[0], X_test.shape[0]), file=f)
        print("\n")
        
        # print("*** Classifier using KNN ***")
        # #build model on knn classifier
        
        
        # # knn = KNeighborsClassifier(n_neighbors=5)
        # # knn.fit(X_train, y_train)
        # # print(knn)
       
        # start = perf_counter()
        # # y_pred = knn.predict(X_test)
        # kfold = KFold(n_splits=10, shuffle=True, random_state=0)
        # classifier = KNeighborsClassifier(n_neighbors=5)
        # scores = cross_val_score(classifier, self.extractedData, self.target, cv=kfold)

        # for fold_index in range(10):
        #     print('In the %i fold, the classification accuracy is %f' %(fold_index+1, scores[fold_index]))
        
        # print('Average classification accuracy is {:.2f}'.format(scores.mean()))
        # end = perf_counter()
        print("\n")

        
        # svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
        
        start = perf_counter()
        # y_pred = svm_model_linear.predict(X_test)
        
        
        
        # end = perf_counter()
        
        # print("naive grid search for best parameters (gamma, C)")
        # X_train, X_test, y_train, y_test = train_test_split(self.extractedData, self.target, random_state=0)
        # print("Size of training set: {} size of test set: {}".format(X_train.shape[0], X_test.shape[0]))
        # best_score = 0
        # for gamma in [0.001, 0.01, 0.1, 1, 10, 100]: 
        #     for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        #         # for each combination of parameters, train an SVC
        #         svm = SVC(gamma=gamma, C=C)
        #         svm.fit(X_train, y_train)
        #         # evaluate the SVC on the test set
        #         score = svm.score(X_test, y_test)
        #         # if we got a better score, store the score and parameters 
        #         if score > best_score:
        #             best_score = score
        #             best_parameters = {'C': C, 'gamma': gamma}
        
        # print("Best score: {:.2f}".format(best_score)) 
        # print("Best parameters: {}".format(best_parameters))

        print("*** GridSearchCV for KNeighborsClassifier \n", file=f)
        # kfold = KFold(n= self.numberOfSegments ,n_folds=10, shuffle=True, random_state=0)

        param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        print("Parameter grid:\n{}".format(param_grid), file=f)
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10)
        X_train, X_test, y_train, y_test = train_test_split(self.extractedData, self.target, random_state=0)
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)), file = f)
        print("Best parameters: {}".format(grid_search.best_params_), file = f)
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_), file = f)
        print("\n")
        print("Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)), file = f)
        print("Confusion matrix:\n{}".format(confusion_matrix(y_test, y_pred)), file = f)
        print(classification_report(y_test, y_pred), file= f)
        print("\n", file=f)

        print("#### Grid Scores ###", file=f)
        #print(grid_search.grid_scores_)

        results = grid_search.grid_scores_
        for i in range(len(results)):
            print(results[i], file=f)



        # result = np.array(list(grid_search.grid_scores_))
        # results = pd.DataFrame(result)
        # print(results)

        #print(grid_search.grid_scores_)


        # convert to DataFrame
        #results = pd.DataFrame(grid_search.cv)
        # show the first 5 rows
        #print(results.head())




    def segment_signal(self, data, window_size): 
        # referenced function meant for inputs
       
       N = data.shape[0]
       dim = data.shape[1]
       K = int(N/window_size)
       
       segments = np.empty((K, window_size, dim))
       
       for i in range(K):
           segment = data[i*window_size:i*window_size+window_size,:]
           segments[i] = np.vstack(segment)
       return segments
       

    def segmentationModule(self, data, frame_size, overlap = False):
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
           overlap_num = 0
           #Count number of valid segments
           while((i + frame_size) < N):

               # start point of overlap
               x = i + int(frame_size / 2)

               #break if overlap exceeds
               if(x + frame_size < N):
                   if (data[x][3] == data[x + frame_size - 1][3]) and (overlap) and (x + frame_size < N):
                       #increment frame number
                       K += 1

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
           self.normData = np.empty((K, frame_size, dim)) #for preprocessing later
                     
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

               #start point of overlap
               x = i + int(frame_size / 2)

               if (x + frame_size < N):
                   if data[x][3] == data[x + frame_size -1][3] and overlap and x + frame_size < N:

                       # sample frame aka segment consists of only up to the 3rd column
                       segment = np.vstack(data[x:x + frame_size, :3])
                       segments[j] = segment

                       # update list of identifiers corresponding to particular sample frame
                       target.append(data[x][3])

                       # move counter for
                       j += 1

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
           self.segmentedData = segments
           
           if self.debug:
               print("Calculated Target frames: " + str(valid_frames) + " Data frames: " + str(actual_frames) + "\n")
               print("Target Data:")
               print(target)
               print("Segmented Data:")
               print(segments[-3:])
           
            
       
    def preprocessingModule(self, data): #preprocessing can be done inline with the segmentation  
       if self.segmentedData is not None and self.numberOfSegments is not None:
           
           #normalize signal data
           #self.norm_data = preprocessing.normalize(self.data[0])
           # imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
           # imp.fit(data)
           #
           # for i in range(self.numberOfSegments):
           #     self.processedData[i] = Imputer(data[i])

           for i in range(self.numberOfSegments):
               self.normData[i] = normalize(data[i])
               
           if self.debug:       
               print("### normalised data set ###")    
               print(self.normData[:3])
           
       else:
           print('data not ready for preprocessing')
           
    def featureExtractionModule(self, data):
       
       
       
       # 12 features will be selected from the accelerometer readings
       featureData = np.empty((self.numberOfSegments, 12))
       
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
           
           #obtain only the first column => accel_x values and etc for accel_y and accel_z
           x_sequence = data[i][:, 0]
           y_sequence = data[i][:, 1]
           z_sequence = data[i][:, 2]
           
           x_mean = np.mean(x_sequence, dtype='Float64')
           y_mean = np.mean(y_sequence, dtype='Float64')
           z_mean = np.mean(z_sequence, dtype='Float64')
           
           x_std = np.std(x_sequence, dtype='Float64')
           y_std = np.std(y_sequence, dtype='Float64')
           z_std = np.std(z_sequence, dtype='Float64')
           
           #obtain only the cov(X, Y) or corr(X, Y) value by the 0th row, 1st column of cov / cor matrix          
           xy_cov = np.cov(x_sequence, y_sequence, ddof=0)[0, 1]
           yz_cov = np.cov(y_sequence, z_sequence, ddof=0)[0, 1]
           zx_cov = np.cov(z_sequence, x_sequence, ddof=0)[0, 1]
           
           xy_corr = np.corrcoef(x_sequence, y_sequence)[0, 1]
           yz_corr = np.corrcoef(y_sequence, z_sequence)[0, 1]
           zx_corr = np.corrcoef(z_sequence, x_sequence)[0, 1]
           
           featureData[i] = np.array([x_mean, y_mean, z_mean, x_std, y_std, z_std, xy_cov, yz_cov, zx_cov, xy_corr, yz_corr, zx_corr])
#           print(x_sequence)
#           print(x_std)
#           print(xy_cov)
#           print(xy_corr)
#           print(featureData[i])
#           break
    
       self.extractedData = featureData
       if self.debug:
           print(featureData[:3])
    
    
    def benchmark(self, start, end, message = False): #used to determine performance of algorithm
        if message and self.debug:
            print(message + '\nTime Elapsed: '+ " %.9f seconds" % (end-start) + '\n')
        else:
            print('\nTime Elapsed: '+ " %.9f seconds" % (end-start) + '\n')

    def save_to_file(self, text):

        with open(self.outputfile, mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(text))
            myfile.write('\n')
   

       