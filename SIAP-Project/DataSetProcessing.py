import pandas as pd
import re
import copy
import numpy as np

def LoadDataSet(fileName):
    df = pd.read_csv(fileName)
    return df

def DropColumns(cols,dataSet):
    dataSet.drop(cols,inplace=True, axis=1)
    return dataSet

def RemoveAllRowsExcept(dataSet, column,value):
    dataSet.drop(dataSet[(dataSet[column] != value)].index, inplace=True)
    return dataSet

def RemoveAllRowsWithNanField(dataSet, column):
    dataSet=dataSet.dropna(subset=[column])
    return dataSet

def FormateImdbScore(dataSet):
    dataSet['IMDb'] = dataSet['IMDb'].apply(lambda x: pd.to_numeric(str(x).split('/')[0]))
    return dataSet

def ExportDataSetAsCsv(dataSet,fileName):
    dataSet.to_csv(fileName+'.csv',index=False)

def MergeTwoDataSetsByColumn(dataSets,cols):
    return pd.merge(dataSets[0], dataSets[1],on=cols, how='outer')

def SortDataSetByColumns(dataSet,columns,ascending):
    dataSet.sort_values(by=columns, ascending = ascending)
    return dataSet


def ChangeNameOfColumn(dataSet,oldName, newName):
    return dataSet.rename(columns={oldName: newName})

def MakeBinaryColumnsForColumn(dataSet,column):
    binTable = pd.get_dummies(dataSet[column])
    dataSet = pd.concat([dataSet,binTable],axis=1,join="inner")
    return dataSet


def ConcatDataSets(dataSets):
    return pd.concat(dataSets,axis=1).reindex(dataSets[0].index)


def ToLowerStringColumnFields(dataSet,column):
        dataSet[column] = dataSet[column].apply(lambda x: str.lower(x))
        return dataSet