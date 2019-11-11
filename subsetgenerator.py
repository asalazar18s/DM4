from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector
import pandas as pd
import numpy as np

# ---------------------- prepare data ----------------------------

# initial 2000x500 dataset
col_names = list(range(0, 500))
filename = "madelon_train.data"
df = pd.read_csv(filename, sep=' ', names=col_names, index_col=False, header=None)

# grab only the columns we want
att_list = [16, 19, 22, 23, 31, 35, 39, 41, 45, 46, 50, 58, 62, 72, 76, 91,
                    92, 100, 105, 106, 107, 120, 122, 123, 125, 138, 141, 142, 145,
                    147, 151, 153, 154, 155, 158, 161, 163, 177, 179, 182, 183, 190,
                    192, 200, 207, 214, 216, 219, 228, 229, 230, 249, 250, 252, 280,
                    283, 295, 301, 302, 305, 309, 317, 337, 353, 360, 362, 372, 393,
                    404, 410, 28, 48, 64, 105, 128, 153, 241, 281, 318, 336, 338, 378,
                    433, 442, 451, 453, 455, 472, 475]

# for col in df.columns:
#     if col not in att_list:
#         df = df.drop(col, axis=1)
# print(df)

# up to this point our Datframe holds only the columns/features we care about.

# we need to get the target values as well.
filename1 = "madelon_train.names"
target_values = pd.read_csv(filename1, sep=' ', index_col=None, header=None)


# up to this point we have the target values as well with shape 1999x1

feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
                                             k_features=20, forward=True,
                                             verbose=2, scoring='accuracy', cv=4)

feature_selector.fit(df, target_values)

print(feature_selector.subsets_)

