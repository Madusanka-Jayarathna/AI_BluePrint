import pandas as pd
from pathlib import Path

def createAndGetPath(basePath:str, stage:str, step:str):
    dirPath =  Path.cwd().parent / basePath / stage / step
    dirPath.mkdir(parents=True, exist_ok=True)
    return dirPath

def saveDataSets(dfTrain:pd.DataFrame, dfVal:pd.DataFrame, dfTest:pd.DataFrame, stage:str, step:str):
    dirPath = createAndGetPath("Data", stage, step)
    dfTrain.to_parquet(dirPath / f"{step}_train.parquet", index=False, engine='fastparquet')
    dfVal.to_parquet(dirPath / f"{step}_val.parquet", index=False, engine='fastparquet')
    dfTest.to_parquet(dirPath / f"{step}_test.parquet", index=False, engine='fastparquet')

    print(f"TrainSet: {dfTrain.shape} | ValSet: {dfVal.shape} | TestSet: {dfTest.shape}")
    print(f"Data sets are succesfully saved : {dirPath}")


def loadDataSets(stage:str, step:str):
    dirPath = createAndGetPath("Data", stage, step)
    dfTrain = pd.read_parquet(dirPath / f"{step}_train.parquet", engine='fastparquet')
    dfVal = pd.read_parquet(dirPath / f"{step}_val.parquet", engine='fastparquet')
    dfTest = pd.read_parquet(dirPath / f"{step}_test.parquet", engine='fastparquet')

    print(f"TrainSet: {dfTrain.shape} | ValSet: {dfVal.shape} | TestSet: {dfTest.shape}")
    return dfTrain, dfVal, dfTest