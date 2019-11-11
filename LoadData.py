import pandas as pd

# Read the train data and set it to a pd DataFrame
col_names = list(range(0,500))
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
    maxVals = data.max(axis=1)
    maxVals = maxVals.max()
    print("Max:")
    print(maxVals)
    return maxVals

def normalize(dataframe):
    #dataframe = dataframe[(dataframe < 650).all(axis=1)]
    result = dataframe
    # print("Data Frame:")
    # print(result)
    max_value = get_max_norm_of_df(result)
    min_value = get_min_value_of_df(result)

    for feature_name in df.columns:
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    result.to_csv(r'./results.csv')
    return result


def correlacion():
    global df
    largest = []

    x = normalize(df)
    z = x.corr(method='pearson')
    z.to_csv(r'./correlated.csv')
    # print("Data Frame Normalized:")
    # print(x)
    # print("Data Frame Correlated:")
    # print(z)

    tuple_list = []
    print("Largest Values: ")
    for i in range(0, 500):
        for j in range(0, 500):
            if abs(z.at[i, j]) <= 0.00002 and z.at[i,j] != 1: # change this to evaluate correlation data
                if z.at[i, j] not in largest:
                    tuple_list.append((i,j, z.at[i,j]))
                    largest.append(z.at[i,j])

    # we have a list of tuples with the format (Column, row, value)
    # the column and row can be interchangeable
    print(len(largest))
    print(len(tuple_list))
    print(tuple_list)
    res_dictionary = {}
    for col, row, val in tuple_list:
        if col in res_dictionary.keys():
            res_dictionary[col].append(row)
        else:
            res_dictionary[col] = []
            res_dictionary[col].append(row)

    print(res_dictionary)
    print(len(res_dictionary))
    col_list = []
    for i in res_dictionary:
        col_list.append(i)
    print(col_list)
    print(len(col_list))
    # remove_cols([319, 443, 452, 473, 379, 337, 456, 476, 129, 339, 282, 434, 242, 494])


def remove_cols(int_array):
    global df

    for val in int_array:
        df = df.drop(val, 1)
    # df.to_csv(r'./df_after_first_corr.csv')
    print(df)

def get_final_df():
    global df

    to_keep_list = [16, 19, 22, 23, 31, 35, 39, 41, 45, 46, 50, 58, 62, 72, 76, 91,
                    92, 100, 105, 106, 107, 120, 122, 123, 125, 138, 141, 142, 145,
                    147, 151, 153, 154, 155, 158, 161, 163, 177, 179, 182, 183, 190,
                    192, 200, 207, 214, 216, 219, 228, 229, 230, 249, 250, 252, 280,
                    283, 295, 301, 302, 305, 309, 317, 337, 353, 360, 362, 372, 393,
                    404, 410, 28, 48, 64, 105, 128, 153, 241, 281, 318, 336, 338, 378,
                    433, 442, 451, 453, 455, 472, 475]

    print(len(to_keep_list))

    for i in range(0, 500):
        if i not in to_keep_list:
            df = df.drop(i, 1)

    df.to_csv(r'./CorrSubset.csv')
    print(df)


get_final_df()
#normalize(df)
#correlacion()