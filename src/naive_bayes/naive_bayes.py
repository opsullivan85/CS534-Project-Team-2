from sklearn.naive_bayes import MultinomialNB
import numpy as np
from src.data_pre_processing import load_data
from src.data_generation import train_path, test_path


def run() -> MultinomialNB:
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)
    y_train = np.minimum(y_train, 1)  # convert all values to 0 or 1
    y_train = y_train.ravel()  # flatten y_train
    y_test = np.minimum(y_test, 1)  # convert all values to 0 or 1
    y_test = y_test.ravel()  # flatten y_train
    X_train += 1  # make all values positive
    X_test += 1  # make all values positive

    mnb = MultinomialNB().fit(X_train, y_train)

    y_pred = mnb.predict(X_train)
    correct = np.sum(y_pred == y_train) / len(y_train)
    false_pos = np.sum((y_pred == 1) & (y_train == 0)) / len(y_train)
    false_neg = np.sum((y_pred == 0) & (y_train == 1)) / len(y_train)
    faulty_correct = np.sum(y_pred[y_train == 1] == y_train[y_train == 1]) / len(
        y_train[y_train == 1]
    )

    print("Train Accuracy: ", correct)
    print("Train False Positive Rate: ", false_pos)
    print("Train False Negative Rate: ", false_neg)
    print("Test-faulty Accuracy: ", faulty_correct)
    print()

    y_pred = mnb.predict(X_test)
    correct = np.sum(y_pred == y_test) / len(y_test)
    false_pos = np.sum((y_pred == 1) & (y_test == 0)) / len(y_test)
    false_neg = np.sum((y_pred == 0) & (y_test == 1)) / len(y_test)
    faulty_correct = np.sum(y_pred[y_test == 1] == y_test[y_test == 1]) / len(
        y_test[y_test == 1]
    )

    print("Test Accuracy: ", correct)
    print("Test False Positive Rate: ", false_pos)
    print("Test False Negative Rate: ", false_neg)
    print("Test-faulty Accuracy: ", faulty_correct)
    print()

    return mnb
