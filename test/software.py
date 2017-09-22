import csv as csv
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
from sklearn import preprocessing
from time import perf_counter, sleep
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split #for splitting training and test data
from sklearn.svm import SVC # Using Linear SVM Classifier
from sklearn.neural_network import MLPClassifier #Feedforward Neural Network Algorithm (Multilayer perceptron)
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

class Software:
    
    #class variables
    fileID = 0
    
    def __init__(self, debug = False):
       Software.fileID += 1
       
       self.debug = debug
       self.rawData = None
       self.segmentedData = None
       self.target = None
       self.extractedData = None
    
    
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
                
        labels = ['Working at Computer', 'Standing Up, Walking and Going up/down stairs', 'Standing', 'Walking', 'Going Up/Down stairs', 'Walking and Talking with Someone', 'Talking while Standing']

        end = perf_counter()
 
        self.benchmark(start, end, '*** Read From CSV ***')
        
        start = perf_counter()
        self.segmentationModule(self.rawData, 120)
        end = perf_counter()
        
        self.benchmark(start, end, '*** Segmentation Module ***')
        # seg = self.segment_signal(self.rawData, 5) #proves that output signal is a 3 dimensional numpy array
        #print(seg[:10])
        
        start = perf_counter()
        self.preprocessingModule(self.segmentedData)
        end = perf_counter()
        
        self.benchmark(start, end, '*** Preprocessing Module ***')
        
        start = perf_counter()
        self.featureExtractionModule(self.segmentedData)
        #self.featureExtractionModule(self.normData)
        end = perf_counter()
        
        self.benchmark(start, end, '*** Feature Extraction Module ***')
        
        # print("size of input: " + str(self.extractedData.shape[0]) + " size of target: "+ str(self.target.shape[0]) + "\n")
    
        
        X = self.extractedData
        y = self.target

        # Split dataset into test data and training data
        start = perf_counter()
        X_train, X_test, y_train, y_test = train_test_split(self.extractedData, self.target, random_state=5)
        end = perf_counter()
        self.benchmark(start, end, '*** Splitting datasets into test and training ***')
        
        # dataset information
        
        print("X_train shape: {}".format(X_train.shape)) 
        print("y_train shape: {}".format(y_train.shape))
        
        print("Size of training set: {} size of test set: {}".format(X_train.shape[0], X_test.shape[0]))
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
        
        # print("\n\n\n")

        # print("naive grid search for best parameters (n_neighbors)")
        # X_train, X_test, y_train, y_test = train_test_split(self.extractedData, self.target, random_state=0)
        # # split data into train+validation set and test set 
        # X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state=0)
        # # split train+validation set into training and validation sets 
        # X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)
        # print("Size of training set: {} size of validation set: {} size of test set:"
        # " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
        # print("Size of training set: {} size of test set: {}".format(X_train.shape[0], X_test.shape[0]))
        # best_score = 0
        # for n_neighbors in range(1,11): 
        #     knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        #     knn.fit(X_train, y_train)
        #     score = knn.score(X_test, y_test)
        #     # if we got a better score, store the score and parameters 
        #     if score > best_score:
        #         best_score = score
        #         best_parameters = n_neighbors
    
        # print("Best score: {:.2f}".format(best_score)) 
        # print("Best parameters: {}".format(best_parameters))      

        
        # print("Test set predictions:\n {}".format(y_pred))
        # print("Test set score: {:.6f}".format(knn.score(X_test, y_test)))
        
        # print("Train set score: {:.6f}".format(knn.score(X_train, y_train)))
        # self.benchmark(start, end, "### KNN Prediction Time ###")
        
        # print("*** 10 fold cross validation ***")
        
        # kfold = KFold(n_splits=10, shuffle=True)
        # fold_index = 0
        # for train, test in kfold.split(X):
        #     model = KNeighborsClassifier(n_neighbors=5).fit(X[train], y[train])
        #     model_predictions = knn.predict(X[test])
        #     accuracy = model.score(X[test], y[test])
        #     cm = confusion_matrix(y[test], model_predictions)
        #     print('In the %i fold, the classification accuracy is %f' %(fold_index, accuracy))
        #     print('And the confusion matrix is: ')
        #     print(cm)
        #     fold_index += 1 
        print("\n")
        print("*** Classifier using Linear SVN ***")
        
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
        

        print("*** Using Scikit GridSearchCV ***")
        # param_grid = [{'kernel': ['rbf'],
        #            'C': [0.001, 0.01, 0.1, 1, 10, 100],
        #            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
        #           {'kernel': ['linear'],'C': [0.001, 0.01, 0.1, 1, 10, 100]}] 
        # param_grid = {'kernel': ['linear'],'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        param_grid = {'kernel': ['linear'],'C': [0.001]}

        # print("List of grids:\n{}".format(param_grid))
        print("Parameter grid:\n{}".format(param_grid))
        grid_search = GridSearchCV(SVC(), param_grid, cv=10)
        X_train, X_test, y_train, y_test = train_test_split(self.extractedData, self.target, random_state=0)
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        print("\n")
        print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
        print("Best parameters: {}".format(grid_search.best_params_))
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
        print("\n")
        print("Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred))) 
        print("Confusion matrix:\n{}".format(confusion_matrix(y_test, y_pred)))        
        print(classification_report(y_test,y_pred))
        print("\n")
        # convert to DataFrame
        results = pd.DataFrame(grid_search.cv_results_) 
        # show the first 5 rows 
        print(results.head())
        
        end = perf_counter()
        print("*** GridSearchCV for KNeighborsClassifier \n")
    
        param_grid = {'n_neighbors': [1,2,3,4,5,6,7,8,9,10]}
        print("Parameter grid:\n{}".format(param_grid))
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10)
        X_train, X_test, y_train, y_test = train_test_split(self.extractedData, self.target, random_state=0)
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
        print("Best parameters: {}".format(grid_search.best_params_))
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
        print("\n")
        print("Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred))) 
        print("Confusion matrix:\n{}".format(confusion_matrix(y_test, y_pred)))        
        print(classification_report(y_test,y_pred))
        print("\n")
        # convert to DataFrame
        results = pd.DataFrame(grid_search.cv_results_) 
        # show the first 5 rows 
        print(results.head())

        # print("Test set predictions:\n {}".format(y_pred))
        # print("Test set score: {:.6f}".format(svm_model_linear.score(X_test, y_test)))
        
        # print("Train set score: {:.6f}".format(svm_model_linear.score(X_train, y_train)))
        # self.benchmark(start, end, "### Linear SVN Prediction Time ###")
        

        # print("*** Classifier using Linear Regression ***")
        # lr = LinearRegression()
        # lr.fit(X_train, y_train)
        
        # start = perf_counter()
        # y_pred = lr.predict(X_test)
        # end = perf_counter()
        
        # # The coefficients
        # print('Coefficients: \n', lr.coef_)
        
        # print("Test set predictions:\n {}".format(y_pred))
        # print("Mean squared error: %.3f " % mean_squared_error(y_test, y_pred))
        # # Explained variance score: 1 is perfect prediction
        # print('Variance score: %.3f ' % r2_score(y_test, y_pred))
        # print("Test set score: {:.6f}".format(lr.score(X_test, y_test)))
        
        # print("Train set score: {:.6f}".format(lr.score(X_train, y_train)))
        # self.benchmark(start, end, "### Linear Regression Prediction Time ###")
        


        # Multi-layer Perceptron classifier.
        # print("*** Classifier using MLPC ***")
        
        # MLPC = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
        # MLPC.fit(X_train, y_train)
        
        # start = perf_counter()
        # y_pred = MLPC.predict(X_test)
        # end = perf_counter()
        # print("Test set predictions:\n {}".format(y_pred))
        # print("Test set score: {:.6f}".format(MLPC.score(X_test, y_test)))
    
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
           
           for i in range(self.numberOfSegments):
               self.normData[i] = preprocessing.normalize(data[i])
               
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
           
           x_mean = np.mean(x_sequence, dtype=np.float64)
           y_mean = np.mean(y_sequence, dtype=np.float64)
           z_mean = np.mean(z_sequence, dtype=np.float64)
           
           x_std = np.std(x_sequence, dtype=np.float64)
           y_std = np.std(y_sequence, dtype=np.float64)
           z_std = np.std(z_sequence, dtype=np.float64)
           
           #obtain only the cov(X, Y) or corr(X, Y) value by the 0th row, 1st column of cov / cor matrix          
           xy_cov = np.cov(x_sequence, y_sequence, ddof=0)[0, 1]
           yz_cov = np.cov(y_sequence, z_sequence, ddof=0)[0, 1]
           zx_cov = np.cov(z_sequence, x_sequence, ddof=0)[0, 1]
           
           xy_corr = np.corrcoef(x_sequence, y_sequence)[0, 1]
           yz_corr = np.corrcoef(y_sequence, z_sequence)[0, 1]
           zx_corr = np.corrcoef(z_sequence, x_sequence)[0, 1]
           
           featureData[i] = np.array([x_mean, y_mean, z_mean, x_std, y_std, z_std, xy_cov, yz_cov, zx_cov, xy_corr, yz_corr, zx_corr], dtype=np.float64)
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

   

       