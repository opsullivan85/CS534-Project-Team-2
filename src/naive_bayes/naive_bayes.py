from sklearn.naive_bayes import GaussianNB
import numpy as np
from src.data_pre_processing import load_data, down_sample_data
from src.data_generation import train_path, test_path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def run() -> GaussianNB:
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)
    y_train = np.minimum(y_train, 1)  # convert all values to 0 or 1
    y_train = y_train.ravel()  # flatten y_train
    y_test = np.minimum(y_test, 1)  # convert all values to 0 or 1
    y_test = y_test.ravel()  # flatten y_train
    X_train += 1  # make all values positive
    X_test += 1  # make all values positive

    X_train, y_train = down_sample_data(X_train, y_train)
    X_test, y_test = down_sample_data(X_test, y_test)

    mnb = GaussianNB().fit(X_train, y_train)
    y_pred = mnb.predict(X_test)

    # Calculate accuracy and confusion matrix for the test set
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return mnb
