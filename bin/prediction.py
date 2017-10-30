import csv as csv
import numpy as np
import pandas as pd
import scipy as sp
import pytz
from datetime import datetime
from sklearn.preprocessing import Imputer, normalize
from time import perf_counter, sleep
from sklearn.cross_validation import train_test_split, KFold #for splitting training and test data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import random


class Software:
    
    #class variables
    fileID = 0
    labels = {0: "Standing",1: "Wave Hands", 2: "Bus Driver", 3: "Front Back", 4: "Sidestep", 5: "Jumping" }
    
    def __init__(self, modelDirectory):
       

       self.classifier = joblib.load(modelDirectory)


    

    def extractFeaturesPredict(self, data):
        # 24 features will be selected from each accelerometer readings
        featureData = np.empty((1, 24))

        # to use the data for classification, we need to convert the data set into a 2D NumPy array

        # Averages
        # 1. Average X    2. Average Y   3. Average Z

        # Standard Deviations
        # 4. Standard deviation X 5. Standard deviation Y 6. Standard deviation Z

        # Covariance
        # 7. Covariance X, Y    8. Covariance Y, Z    9. Covariance Z, X

        # Correlation
        # 10. Correlation X, Y   11. Correlation Y, Z    12. Correlation Z, X

        # obtain only the first column => accel_x values and etc for accel_y and accel_z
        x_sequence = data[:, 0]
        y_sequence = data[:, 1]
        z_sequence = data[:, 2]

        a_sequence = data[:, 3]
        b_sequence = data[:, 4]
        c_sequence = data[:, 5]

        x_mean = np.mean(x_sequence, dtype='Int32')
        y_mean = np.mean(y_sequence, dtype='Int32')
        z_mean = np.mean(z_sequence, dtype='Int32')

        a_mean = np.mean(a_sequence, dtype='Int32')
        b_mean = np.mean(b_sequence, dtype='Int32')
        c_mean = np.mean(c_sequence, dtype='Int32')

        x_std = np.std(x_sequence, dtype='Int32')
        y_std = np.std(y_sequence, dtype='Int32')
        z_std = np.std(z_sequence, dtype='Int32')

        a_std = np.std(a_sequence, dtype='Int32')
        b_std = np.std(b_sequence, dtype='Int32')
        c_std = np.std(c_sequence, dtype='Int32')

        # obtain only the cov(X, Y) or corr(X, Y) value by the 0th row, 1st column of cov / cor matrix
        xy_cov = np.cov(x_sequence, y_sequence, ddof=0)[0, 1]
        yz_cov = np.cov(y_sequence, z_sequence, ddof=0)[0, 1]
        zx_cov = np.cov(z_sequence, x_sequence, ddof=0)[0, 1]

        ab_cov = np.cov(a_sequence, b_sequence, ddof=0)[0, 1]
        bc_cov = np.cov(b_sequence, c_sequence, ddof=0)[0, 1]
        ca_cov = np.cov(c_sequence, a_sequence, ddof=0)[0, 1]

        xy_corr = np.corrcoef(x_sequence, y_sequence)[0, 1]
        yz_corr = np.corrcoef(y_sequence, z_sequence)[0, 1]
        zx_corr = np.corrcoef(z_sequence, x_sequence)[0, 1]

        ab_corr = np.corrcoef(a_sequence, b_sequence)[0, 1]
        bc_corr = np.corrcoef(b_sequence, c_sequence)[0, 1]
        ca_corr = np.corrcoef(c_sequence, a_sequence)[0, 1]

        featureData = np.array(
            [x_mean, y_mean, z_mean, x_std, y_std, z_std, xy_cov, yz_cov, zx_cov, xy_corr, yz_corr, zx_corr, a_mean,
             b_mean, c_mean, a_std, b_std, c_std, ab_cov, bc_cov, ca_cov, ab_corr, bc_corr, ca_corr])

        #correct data to fix incorrect values

        index = 0
        for j in featureData:
            # print(i)
            if not np.isfinite(j) or np.isnan(j):

                featureData = np.nan_to_num(featureData)
            index += 1

        return featureData

    def extractFeaturesPredictNew(self, data):
        # 24 features will be selected from each accelerometer readings
        featureData = np.empty((1, 24))

        # to use the data for classification, we need to convert the data set into a 2D NumPy array

        # Averages
        # 1. Average X    2. Average Y   3. Average Z

        # Standard Deviations
        # 4. Standard deviation X 5. Standard deviation Y 6. Standard deviation Z

        # Covariance
        # 7. Covariance X, Y    8. Covariance Y, Z    9. Covariance Z, X

        # Correlation
        # 10. Correlation X, Y   11. Correlation Y, Z    12. Correlation Z, X

        # obtain only the first column => accel_x values and etc for accel_y and accel_z
        x_sequence = data[:, 0]
        y_sequence = data[:, 1]
        z_sequence = data[:, 2]

        a_sequence = data[:, 3]
        b_sequence = data[:, 4]
        c_sequence = data[:, 5]

        x_mean = np.mean(x_sequence, dtype='Int32')
        y_mean = np.mean(y_sequence, dtype='Int32')
        z_mean = np.mean(z_sequence, dtype='Int32')

        a_mean = np.mean(a_sequence, dtype='Int32')
        b_mean = np.mean(b_sequence, dtype='Int32')
        c_mean = np.mean(c_sequence, dtype='Int32')

        x_std = np.std(x_sequence, dtype='Int32')
        y_std = np.std(y_sequence, dtype='Int32')
        z_std = np.std(z_sequence, dtype='Int32')

        a_std = np.std(a_sequence, dtype='Int32')
        b_std = np.std(b_sequence, dtype='Int32')
        c_std = np.std(c_sequence, dtype='Int32')

        x_min = np.nanmin(x_sequence)
        y_min = np.nanmin(y_sequence)
        z_min = np.nanmin(z_sequence)

        a_min = np.nanmin(a_sequence)
        b_min = np.nanmin(b_sequence)
        c_min = np.nanmin(c_sequence)

        x_max = np.nanmax(x_sequence)
        y_max = np.nanmax(y_sequence)
        z_max = np.nanmax(z_sequence)

        a_max = np.nanmax(a_sequence)
        b_max = np.nanmax(b_sequence)
        c_max = np.nanmax(c_sequence)

        # obtain only the cov(X, Y) or corr(X, Y) value by the 0th row, 1st column of cov / cor matrix
        xy_cov = np.cov(x_sequence, y_sequence, ddof=0)[0, 1]
        yz_cov = np.cov(y_sequence, z_sequence, ddof=0)[0, 1]
        zx_cov = np.cov(z_sequence, x_sequence, ddof=0)[0, 1]

        ab_cov = np.cov(a_sequence, b_sequence, ddof=0)[0, 1]
        bc_cov = np.cov(b_sequence, c_sequence, ddof=0)[0, 1]
        ca_cov = np.cov(c_sequence, a_sequence, ddof=0)[0, 1]

        xy_corr = np.corrcoef(x_sequence, y_sequence)[0, 1]
        yz_corr = np.corrcoef(y_sequence, z_sequence)[0, 1]
        zx_corr = np.corrcoef(z_sequence, x_sequence)[0, 1]

        ab_corr = np.corrcoef(a_sequence, b_sequence)[0, 1]
        bc_corr = np.corrcoef(b_sequence, c_sequence)[0, 1]
        ca_corr = np.corrcoef(c_sequence, a_sequence)[0, 1]

        featureData = np.array(
            [x_mean, y_mean, z_mean, a_mean, b_mean, c_mean,
             x_std, y_std, z_std, a_std, b_std, c_std,
             xy_cov, yz_cov, zx_cov, ab_cov, bc_cov, ca_cov,
             xy_corr, yz_corr, zx_corr,  ab_corr, bc_corr, ca_corr,
             x_min, y_min, z_min, a_min, b_min, c_min,
             x_max, y_max, z_max, a_max, b_max, c_max])

        #correct data to fix incorrect values

        index = 0
        for j in featureData:
            # print(i)
            if not np.isfinite(j) or np.isnan(j):
                featureData = np.nan_to_num(featureData)

            index += 1

        return featureData

    def predictDanceMove(self, signal):
        try:
            rawSignal = np.asarray(signal)
            processedSignal = self.extractFeaturesPredict(rawSignal)
            prediction_result = self.classifier.predict(processedSignal)

            return Software.labels[prediction_result[0]]
        except:
            return "Error"


       