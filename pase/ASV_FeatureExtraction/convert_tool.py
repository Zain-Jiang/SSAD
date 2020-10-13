#!/bin/bash
# -*- coding:utf-8 -*-

import numpy as np

def cell2GMM(cellData):
    '''
    Function for converting cell feature to GMM data format

    Input:
    cellData: cell feature data which time along the row, feature along the columns. (1 * dimension) shape like

    Output: array of GMM data format
            which has fixed columns and combine the time row.

    Written by Lizetian
    '''

    col,row = cellData.shape
    gmmData = cellData[0,0]
    for i in range(1,row):
        gmmData = np.concatenate((gmmData,cellData[0,i]),axis=1)
    return gmmData

