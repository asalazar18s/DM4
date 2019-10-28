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
    #print(z.nlargest(1,154)[1])
    # for i in range(1,500):
    #     largest.append(z.nlargest(1, i).index)
    #
    # lar = largest.sort()
    # for i in range(0,20):
    #     print(lar)
    #print(z.nsmallest(10,1)[1])
    print("Largest Values: ")
    for i in range(1, 500):
        for j in range(1, 500):
            if abs(z.at[i,j]) > 0.6 and z.at[i,j] != 1:
                print(z.at[i,j])
                largest.append(z.at[i,j])

    print(len(largest))

covarianza()
