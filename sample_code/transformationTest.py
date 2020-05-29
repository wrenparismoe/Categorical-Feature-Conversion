from sklearn.metrics import *


def test(x_train, x_test, y_train, y_test, transformations, tnames, classifier):
    count = 0
    for encoder in transformations:
        encoder.fit(x_train)
        x_train_enc = encoder.transform(x_train)
        x_test_enc = encoder.transform(x_test)

        clf = classifier
        clf.fit(x_train_enc, y_train)

        y_pred = clf.predict(x_test_enc)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        print(tnames[count])

        print("ACCURACY:", acc)
        print("PRECISION:", prec)
        print("RECALL:", rec, '\n')
        count += 1
