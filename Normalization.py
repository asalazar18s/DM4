import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, RadiusNeighborsClassifier
from sklearn import metrics

# Read the train data and set it to a pd DataFrame
col_names = list(range(0, 500))
filename = "madelon_train.data"
df = pd.read_csv(filename, sep=' ', names=col_names, index_col=False, header=None)

filename1 = "madelon_train.names"
training_target_values = pd.read_csv(filename1, sep=' ', index_col=None, header=None)

# Open validation data
validation_file_name = "madelon_valid.data"
validation_data = pd.read_csv(validation_file_name, sep=' ', names=col_names, index_col=False, header=None)
filename2 = "madelon_valid.labels"
validation_target_values = pd.read_csv(filename2, sep=' ', index_col=None, header=None)

# Open validation data
test_file_name = "madelon_test.data"
test_data = pd.read_csv(test_file_name, sep=' ',names=col_names, index_col=False, header=None)

best_attributes = [28, 90, 153, 288, 292, 318, 378, 455, 197, 475, 142, 100, 451, 472, 128, 7, 151, 468, 482, 3, 0]


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
    result = dataframe
    # max_value = get_max_norm_of_df(result)
    # min_value = get_min_value_of_df(result)
    max_value = 999
    min_value = 52

    for feature_name in result.columns:
        result[feature_name] = (result[feature_name] - min_value) / (max_value - min_value)

    result.to_csv(r'./results.csv')
    return result

#--------------------------------------------------------------------------------------------------------------

def normalize_train_df():
    global df

    print("Train Dataframe")
    for col in df.columns:
        if col not in best_attributes:
            df = df.drop(col, axis=1)

    print("Pre Normalization Train")
    print(df)

    print("\n\nPost Normalization Train")
    print(normalize(df), "\n\n\n\n\n\n\n")
    return normalize(df)


def normalize_validation_df():
    global validation_data
    print("Validation Dataframe")
    for col in validation_data.columns:
        if col not in best_attributes:
            validation_data = validation_data.drop(col, axis=1)

    print("Pre Normalization Validation")
    print(validation_data)

    print("\n\nPost Normalization Validation")
    print(normalize(validation_data), "\n\n\n\n\n\n\n")
    return normalize(validation_data)


def normalize_test_df():
    global test_data

    for col in test_data.columns:
        if col not in best_attributes:
            test_data = test_data.drop(col, axis=1)

    # print("Pre Normalization Test")
    # print(test_data)
    #
    # print("\n\nPost Normalization Test")
    # print(normalize(test_data), "\n\n\n\n\n\n\n")
    # return(normalize(test_data))

#-------------------------------------------------------------------------------------------------------------
train_norm = normalize_train_df()
valid_norm = normalize_validation_df()
# test_norm = normalize_test_df()

def knn_training_testing(neighbor):
    '''

    :param m: metric to use 'eucliden', 'manhattan'
    :return:
    '''
    global training_target_values, train_norm , validation_target_values, valid_norm

    neighbors = neighbor

    # train the knn algorithm p=1 euclidean, p=2 manhattan
    knn = KNeighborsClassifier(n_neighbors=neighbors,
                               weights='uniform',
                               metric='euclidean',

                               )

    knn.fit(train_norm, training_target_values)
    print("Neighbors: ", neighbors)
    print("Accuracy: ", knn.score(X=valid_norm, y=validation_target_values))
    return(neighbors, knn.score(X=valid_norm, y=validation_target_values))

tuple_list = []
for i in range(1,31):
    tuple_list.append(knn_training_testing(i))

print(tuple_list)

