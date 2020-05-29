from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



pd.options.display.max_columns = 999
pd.options.display.width = 999

# Learning curve - graph where x-axis is # of data points and y-axis is score

# Use pipelines to combine all steps in one
# Use transformations to do personalized encoding in less steps - Pipeline, make_pipeline
    # ColumnTransformer, make_column_transformer to do personalized encoding for each column
# Grid Search to find best parameters for a given method - use pipelines
# model complexity  - for min_samples_split


def load_dataset():
    filePath = "C:/Users/wrenp/Documents/Spring 2020/Mathematical Modeling/processed_data/adult/"
    x_train = pd.read_csv(filePath + "x_train.csv")
    x_test = pd.read_csv(filePath + "x_test.csv")
    y_train = pd.read_csv(filePath + "y_train.csv")
    y_test = pd.read_csv(filePath + "y_test.csv")
    featureList = []
    for i in range(len(x_train.columns)):
        featureList.append("x" + str(i))
    x_train.columns = featureList
    x_test.columns = featureList
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_dataset()


# y_train = y_train.values.ravel()
# y_test = y_test.values.ravel()

pipeline = Pipeline([('enc', OneHotEncoder(handle_unknown='ignore')), ('clf', DecisionTreeClassifier(criterion='entropy', min_samples_split=200, max_depth=24))])

pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)



# USE FOR CLASSIFICATION GRID SEARCH
"""

pipeline = Pipeline([('oh', OneHotEncoder(handle_unknown='ignore')), ('dt', DecisionTreeClassifier(criterion='entropy', min_samples_split=200, max_depth=24))])

# pipeline = Pipeline([('enc', OneHotEncoder(handle_unknown='ignore')), ('clf', RandomForestClassifier())])
# Create lists of parameter for Decision Tree Classifier
n_estimators = list(range(200, 1001, 200))
# list(range(10,120,20))

# Create a dictionary of all the parameter options
# Note has you can access the parameters of steps of a pipeline by using '__’

parameters = dict(clf__n_estimators=n_estimators)

clf = GridSearchCV(pipeline, parameters, n_jobs=-1)
# clf = RandomizedSearchCV(pipeline, parameters, n_jobs=-1)
clf.fit(x_train, y_train)

best = clf.best_estimator_.get_params()

for p in parameters:
    print('Best ', str(p)[5:], ':', best[p])
###
print()
y_pred = clf.predict(x_test)
"""


# PIPELINE WITHOUT GRID SEARCH
# pipeline.fit(x_train, y_train)
# x_train = pipeline.transform(x_train)
# y_pred = pipeline.predict(x_test)


acc = metrics.accuracy_score(y_test, y_pred)
prec = metrics.precision_score(y_test, y_pred)
rec = metrics.recall_score(y_test, y_pred)
print("ACCURACY:", acc)
print("PRECISION:", prec)
print("RECALL:", rec)

report = metrics.classification_report(y_test,y_pred, digits=10)

scores = cross_val_score(pipeline, x_train, y_train, cv=5, n_jobs=-1)

print('CLASSIFICATION REPORT: \n', report, '\n')
print('CROSS VALIDATION SCORES: \n', scores)





#
# title = "Learning Curves (Decision Tree)"
#
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
#
# estimator = DecisionTreeClassifier()
# plot_learning_curve.plot_learning_curve(estimator, title, x_train, y_train, axes=None, ylim=(0.7, 1.01),
#                     cv=cv, n_jobs=4)


