import pandas as pd

def split(path,target, sname):
    df = pd.read_excel(path, sheet_name = sname)
    df = df.dropna(axis=1)
    x = df.drop([target], axis = 1)
    y = df[target]
    return x, y
