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
    labels = {0: "standing",1: "wavehands", 2: "busdriver", 3: "frontback", 4: "sidestep", 5: "jumping", 6: "jumpingjack", 7: "turnclap", 8: "squatturnclap", 9: "windowcleaner", 10: "windowcleaner360", 11: "logout" }
    
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

    def extractFeaturesPredict6(self, data):
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

    def extractFeaturesPredict12(self, data):
        # 24 features will be selected from each accelerometer readings
        featureData = np.empty((1, 72))

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
        u_sequence = data[:, 0]
        v_sequence = data[:, 1]
        w_sequence = data[:, 2]

        x_sequence = data[:, 3]
        y_sequence = data[:, 4]
        z_sequence = data[:, 5]

        a_sequence = data[:, 6]
        b_sequence = data[:, 7]
        c_sequence = data[:, 8]

        d_sequence = data[:, 9]
        e_sequence = data[:, 10]
        f_sequence = data[:, 11]

        u_mean = np.mean(u_sequence, dtype='Int32')
        v_mean = np.mean(v_sequence, dtype='Int32')
        w_mean = np.mean(w_sequence, dtype='Int32')

        x_mean = np.mean(x_sequence, dtype='Int32')
        y_mean = np.mean(y_sequence, dtype='Int32')
        z_mean = np.mean(z_sequence, dtype='Int32')

        a_mean = np.mean(a_sequence, dtype='Int32')
        b_mean = np.mean(b_sequence, dtype='Int32')
        c_mean = np.mean(c_sequence, dtype='Int32')

        d_mean = np.mean(d_sequence, dtype='Int32')
        e_mean = np.mean(e_sequence, dtype='Int32')
        f_mean = np.mean(f_sequence, dtype='Int32')

        u_std = np.std(u_sequence, dtype='Int32')
        v_std = np.std(v_sequence, dtype='Int32')
        w_std = np.std(w_sequence, dtype='Int32')

        x_std = np.std(x_sequence, dtype='Int32')
        y_std = np.std(y_sequence, dtype='Int32')
        z_std = np.std(z_sequence, dtype='Int32')

        a_std = np.std(a_sequence, dtype='Int32')
        b_std = np.std(b_sequence, dtype='Int32')
        c_std = np.std(c_sequence, dtype='Int32')

        d_std = np.std(d_sequence, dtype='Int32')
        e_std = np.std(e_sequence, dtype='Int32')
        f_std = np.std(f_sequence, dtype='Int32')

        u_min = np.nanmin(u_sequence)
        v_min = np.nanmin(v_sequence)
        w_min = np.nanmin(w_sequence)

        x_min = np.nanmin(x_sequence)
        y_min = np.nanmin(y_sequence)
        z_min = np.nanmin(z_sequence)

        a_min = np.nanmin(a_sequence)
        b_min = np.nanmin(b_sequence)
        c_min = np.nanmin(c_sequence)

        d_min = np.nanmin(d_sequence)
        e_min = np.nanmin(e_sequence)
        f_min = np.nanmin(f_sequence)

        u_max = np.nanmax(u_sequence)
        v_max = np.nanmax(v_sequence)
        w_max = np.nanmax(w_sequence)

        x_max = np.nanmax(x_sequence)
        y_max = np.nanmax(y_sequence)
        z_max = np.nanmax(z_sequence)

        a_max = np.nanmax(a_sequence)
        b_max = np.nanmax(b_sequence)
        c_max = np.nanmax(c_sequence)

        d_max = np.nanmax(d_sequence)
        e_max = np.nanmax(e_sequence)
        f_max = np.nanmax(f_sequence)

        # obtain only the cov(X, Y) or corr(X, Y) value by the 0th row, 1st column of cov / cor matrix
        uv_cov = np.cov(u_sequence, v_sequence, ddof=0)[0, 1]
        vw_cov = np.cov(v_sequence, w_sequence, ddof=0)[0, 1]
        wu_cov = np.cov(w_sequence, u_sequence, ddof=0)[0, 1]

        xy_cov = np.cov(x_sequence, y_sequence, ddof=0)[0, 1]
        yz_cov = np.cov(y_sequence, z_sequence, ddof=0)[0, 1]
        zx_cov = np.cov(z_sequence, x_sequence, ddof=0)[0, 1]

        ab_cov = np.cov(a_sequence, b_sequence, ddof=0)[0, 1]
        bc_cov = np.cov(b_sequence, c_sequence, ddof=0)[0, 1]
        ca_cov = np.cov(c_sequence, a_sequence, ddof=0)[0, 1]

        de_cov = np.cov(d_sequence, e_sequence, ddof=0)[0, 1]
        ef_cov = np.cov(e_sequence, f_sequence, ddof=0)[0, 1]
        fd_cov = np.cov(f_sequence, d_sequence, ddof=0)[0, 1]

        uv_corr = np.corrcoef(u_sequence, v_sequence)[0, 1]
        vw_corr = np.corrcoef(v_sequence, w_sequence)[0, 1]
        wu_corr = np.corrcoef(w_sequence, u_sequence)[0, 1]

        xy_corr = np.corrcoef(x_sequence, y_sequence)[0, 1]
        yz_corr = np.corrcoef(y_sequence, z_sequence)[0, 1]
        zx_corr = np.corrcoef(z_sequence, x_sequence)[0, 1]

        ab_corr = np.corrcoef(a_sequence, b_sequence)[0, 1]
        bc_corr = np.corrcoef(b_sequence, c_sequence)[0, 1]
        ca_corr = np.corrcoef(c_sequence, a_sequence)[0, 1]

        de_corr = np.corrcoef(d_sequence, e_sequence)[0, 1]
        ef_corr = np.corrcoef(e_sequence, f_sequence)[0, 1]
        fd_corr = np.corrcoef(f_sequence, d_sequence)[0, 1]

        featureData = np.array(
            [u_mean, v_mean, w_mean, x_mean, y_mean, z_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean,
             u_std, v_std, w_std, x_std, y_std, z_std, a_std, b_std, c_std, d_std, e_std, f_std,
             uv_cov, vw_cov, wu_cov, xy_cov, yz_cov, zx_cov, ab_cov, bc_cov, ca_cov, de_cov, ef_cov, fd_cov,
             uv_corr, vw_corr, wu_corr, xy_corr, yz_corr, zx_corr, ab_corr, bc_corr, ca_corr, de_corr, ef_corr, fd_corr,
             u_min, v_min, w_min, x_min, y_min, z_min, a_min, b_min, c_min, d_min, e_min, f_min,
             u_max, v_max, w_max, x_max, y_max, z_max, a_max, b_max, c_max, d_max, e_max, f_max])

        #correct data to fix incorrect values


        for j in featureData:
            # print(i)
            if not np.isfinite(j) or np.isnan(j):

                featureData = np.nan_to_num(featureData)



        return featureData

    def predictDanceMove(self, signal):
        
            rawSignal = np.asarray(signal)
            processedSignal = self.extractFeaturesPredict12(rawSignal)
            prediction_result = self.classifier.predict(processedSignal)

            return Software.labels[prediction_result[0]]
        #except:
            #return "Error"


       
