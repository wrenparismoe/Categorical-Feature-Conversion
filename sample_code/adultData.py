import pandas as pd

pd.options.display.max_columns = 999
pd.options.display.width = 999

def load_dataset():
    filePath = "C:/Users/wrenp/Documents/Spring 2020/Mathematical Modeling/processed_data/adult/"
    x_train = pd.read_csv(filePath + "x_train.csv")
    x_test = pd.read_csv(filePath + "x_test.csv")
    y_train = pd.read_csv(filePath + "y_train.csv")
    y_test = pd.read_csv(filePath + "y_test.csv")
    x_names = pd.read_csv(filePath + "x_names.csv")
    featureList = list(x_names)
    x_train.columns = featureList
    x_test.columns = featureList
    return x_train, x_test, y_train, y_test
