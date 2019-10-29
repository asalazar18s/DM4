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
    maxVals = maxVals.max()
    print("Max:")
    print(maxVals)
    return maxVals

def normalize(dataframe):
    #dataframe = dataframe[(dataframe < 650).all(axis=1)]
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
    largest = []

    x = normalize(df)
    z = x.corr(method='pearson')
    z.to_csv(r'./correlated.csv')
    print("Data Frame Normalized:")
    print(x)
    print("Data Frame Correlated:")
    print(z)

    tuple_list = []
    print("Largest Values: ")
    for i in range(1, 500):
        for j in range(1, 500):
            if abs(z.at[i,j]) > 0.6 and z.at[i,j] != 1:
                if z.at[i,j] not in largest:
                    tuple_list.append((i,j, z.at[i,j]))
                    largest.append(z.at[i,j])

    # we have a list of tuples with the format (Column, row, value)
    # the column and row can be interchangable
    print(len(largest))
    print(tuple_list)



covarianza()
