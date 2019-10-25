import pandas as pd

# Read the train data and set it to a pd DataFrame
col_names = list(range(1,501))
filename = "madelon_train.data"
df = pd.read_csv(filename, sep=' ', names=col_names, index_col=False, header=None)
#print(df)

def get_min_value_of_df(data):
    global df
    minVals = data.min(axis = 1)
    minVals = minVals.min()
    print("Min:")
    print(minVals)
    return minVals

def get_max_norm_of_df(data):
    global df
    maxVals = data.max(axis = 1)
    maxVals = maxVals.min()
    print("Max:")
    print(maxVals)
    return maxVals

def normalize(dataframe):
    dataframe = dataframe[(dataframe < 600).all(axis=1)]
    result = dataframe
    print("Data Frame:")
    print(result)
    max_value = get_max_norm_of_df(result)
    min_value = get_min_value_of_df(result)

    for feature_name in df.columns:
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    result.to_csv(r'./results.csv')
    return result

def covarianza():
    global df

    x = normalize(df)
    z = x.corr()
    print("Data Frame Normalized:")
    print(x)
    print("Data Frame Correlated:")
    print(z)
    print(z.nlargest(10,1)[1])

covarianza()
