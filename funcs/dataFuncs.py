
import pandas as pd


def loadCSV_data(csvFile, topRows=-1):
    if topRows < 0:
        df = pd.read_csv(csvFile, header=0, sep=',')
    else:
        df = pd.read_csv(csvFile, header=0, sep=',', nrows=topRows)

    for idxNum, column in enumerate(df.columns):
        df = df.rename(columns={column: (column+'lxx'+str(idxNum))})

    return df


def delColBasedonMarkNum(df, drops):
    for delNum in drops:
        for column in df.columns:
            sigStr = 'lxx'+str(delNum)
            if sigStr in column:
                df = df.drop(column, axis=1)
                continue

    return df


def replaceColNums(df, digitN):
    dfCook = df
    newCols = []
    for column in dfCook.columns:
        newCols.append(column[-digitN:])
    dfCook.columns = newCols
    return dfCook
