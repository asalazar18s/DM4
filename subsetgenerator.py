from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# ---------------------- prepare data ----------------------------

# initial 2000x500 dataset
col_names = list(range(0, 500))
filename = "madelon_train.data"
df = pd.read_csv(filename, sep=' ', names=col_names, index_col=False, header=None)

# grab only the columns we want
att_list = [0, 4, 6, 7, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 25,
            28, 29, 30, 31, 33, 35, 36, 37, 41, 42, 43, 44, 46, 47, 52, 53, 54,
            55, 56, 57, 58, 59, 62, 63, 65, 66, 67, 69, 70, 72, 73, 74, 75, 76,
            78, 79, 80, 81, 83, 84, 86, 87, 90, 92, 94, 95, 96, 97, 99, 101, 102,
            104, 105, 106, 107, 109, 111, 112, 114, 116, 117, 118, 120, 121, 122,
            123, 124, 126, 129, 130, 132, 133, 134, 137, 138, 139, 141, 144, 145,
            146, 147, 148, 149, 150, 151, 152, 153, 154, 157, 158, 159, 160, 161,
            162, 164, 165, 166, 171, 176, 177, 178, 179, 180, 182, 184, 186, 187,
            188, 189, 192, 193, 195, 198, 199, 200, 202, 203, 204, 205, 206, 207,
            211, 212, 216, 217, 218, 221, 222, 224, 225, 226, 227, 228, 229, 230,
            231, 232, 233, 235, 236, 237, 240, 243, 244, 246, 247, 248, 249, 252,
            255, 257, 258, 260, 261, 263, 264, 266, 267, 268, 270, 276, 284, 285,
            286, 288, 289, 290, 297, 300, 301, 302, 303, 306, 310, 316, 323, 324,
            325, 326, 331, 332, 334, 335, 336, 343, 344, 347, 350, 355, 356, 357,
            365, 367, 370, 372, 378, 385, 389, 402, 403, 438, 444, 450, 456, 489]


    # [16, 19, 22, 23, 31, 35, 39, 41, 45, 46, 50, 58, 62, 72, 76, 91,
    #                 92, 100, 105, 106, 107, 120, 122, 123, 125, 138, 141, 142, 145,
    #                 147, 151, 153, 154, 155, 158, 161, 163, 177, 179, 182, 183, 190,
    #                 192, 200, 207, 214, 216, 219, 228, 229, 230, 249, 250, 252, 280,
    #                 283, 295, 301, 302, 305, 309, 317, 337, 353, 360, 362, 372, 393,
    #                 404, 410, 28, 48, 64, 105, 128, 153, 241, 281, 318, 336, 338, 378,
    #                 433, 442, 451, 453, 455, 472, 475]

# for col in df.columns:
#     if col not in att_list:
#         df = df.drop(col, axis=1)
# print(df)

# up to this point our Datframe holds only the columns/features we care about.

# we need to get the target values as well.
filename1 = "madelon_train.names"
target_values = pd.read_csv(filename1, sep=' ', index_col=None, header=None)


# up to this point we have the target values as well with shape 1999x1

#feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
#                                             k_features=20, forward=True,
#                                             verbose=2, scoring='accuracy', cv=4)

neighbors = 9
# train the knn algorithm
knn = KNeighborsClassifier(n_neighbors=neighbors, metric='minkowski')
feature_selector = SequentialFeatureSelector(knn,
                                             k_features=20,
                                             forward=True,
                                             verbose=2,
                                             scoring='accuracy',
                                             cv=4)

feature_selector.fit(df, target_values)

print(feature_selector.subsets_)

