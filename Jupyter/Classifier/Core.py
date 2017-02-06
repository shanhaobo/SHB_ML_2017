# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 00:07:53 2017

@author: ShanHaoBo
"""

__all__ = ["Sigmoid", "Gradient", "Cost"]

import numpy as np

def Sigmoid(inData) :
    return 1 / (1 + np.exp(-inData))

def Gradient(inData, inWeights, inLabels, inActiveFunc=Sigmoid) :
    tM = 1 / len(inLabels)
    tPred = inActiveFunc(inData.dot(inWeights))
    tGradient = tM * inData.T.dot(tPred - inLabels)
    return tGradient

def Cost(inData, inWeights, inLabels, inActiveFunc=Sigmoid) :
    tM = 1 / len(inLabels)
    tPred = inActiveFunc(inData.dot(inWeights))
    tCost = tM * np.sum(-inLabels * np.log(tPred) - (1 - inLabels) * np.log(1 - tPred))
    return tCost
