# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:41:30 2017

@author: Administrator
"""

import csv
import numpy as np

class __IrisDataLoaderClass:
    
    # raw data
    RawData = None
    RawDataLabels = None
    
    @classmethod
    def LoadRawData(cls):
        if cls.RawData is None:
            # read raw data from csv file
            with open("..\\Data\\iris-species\\iris.csv") as tTempFile:
                tCSVReader = csv.reader(tTempFile)
                tRawData = [row for row in tCSVReader]

            # remove head row
            tRawData_1 = np.array(tRawData[1:])
            
            DataRowCnt, DummyColCnt = tRawData_1.shape
            
            # init label
            tDataLabels = np.zeros(DataRowCnt, float)
            
            # generate label
            for i in range(DataRowCnt):
                if tRawData_1[i][-1] == 'Iris-setosa':
                    tDataLabels[i] = 1
                elif tRawData_1[i][-1] == 'Iris-versicolor':
                    tDataLabels[i] = 2
                elif tRawData_1[i][-1] == 'Iris-virginica':
                    tDataLabels[i] = 3

            # member variant
            cls.RawDataLabels = tDataLabels.reshape((DataRowCnt, 1))
            
            # remove index col(0) & label col(-1)
            tRawData_2 = tRawData_1[:, 1:-1]
    
            # convert to float type
            cls.RawData = tRawData_2.astype(float)
            

    def __init__(self):
        __IrisDataLoaderClass.LoadRawData()
        

    def DataWithBias(self, inBias):
        tBias = np.full((len(__IrisDataLoaderClass.RawDataLabels), 1), inBias, float)
        return np.column_stack((tBias, __IrisDataLoaderClass.RawData))
    
def DataWithBias_Labels(inBias):
    ic = __IrisDataLoaderClass()
    return ic.DataWithBias(inBias), __IrisDataLoaderClass.RawDataLabels
