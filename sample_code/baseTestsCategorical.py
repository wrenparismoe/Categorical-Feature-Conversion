from modules.adultData import load_dataset
from modules.transformationTest import test
from category_encoders import *
from sklearn.tree import DecisionTreeClassifier
import numpy as np


x_train, x_test, y_train, y_test = load_dataset()

x_train = x_train.select_dtypes(include=[np.number]).astype(float).join(x_train.select_dtypes(exclude=[np.number]))
x_test = x_test.select_dtypes(include=[np.number]).astype(float).join(x_test.select_dtypes(exclude=[np.number]))


features = list(x_train.columns)
numerical_features = list(x_train.select_dtypes(include=[float]).columns)
categorical_features = [x for x in x_train.columns if x not in numerical_features]


oh = OneHotEncoder(return_df=True)


ordinal = OrdinalEncoder(return_df=True)

BDE = BackwardDifferenceEncoder(return_df=True)

base1 = BaseNEncoder(return_df=True, base=1)
base2 = BaseNEncoder(return_df=True, base=2)
base8 = BaseNEncoder(return_df=True, base=8)
base10 = BaseNEncoder(return_df=True, base=10)
base16 = BaseNEncoder(return_df=True, base=16)


binary = BinaryEncoder(return_df=True)

import multiprocessing
hashing8 = HashingEncoder(return_df=True, max_process=10, n_components=8)
hashing16 = HashingEncoder(return_df=True, max_process=2, n_components=16)
hashing32 = HashingEncoder(return_df=True, max_process=2, n_components=32)




transformations = [oh, ordinal, BDE]
tnames = ['One-hot', 'ordinal', 'Backwards Difference']

# transformations = [hashing8,hashing16,hashing32]
# tnames = ['Hash 8', 'Hash 16', 'Hash 32']

test(x_train, x_test, y_train, y_test, transformations, tnames, DecisionTreeClassifier(criterion='entropy', max_depth=24, min_samples_split=200))
