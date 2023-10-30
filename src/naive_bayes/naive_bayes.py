from sklearn import naive_bayes as nb
import numpy as np
from src.data_pre_processing import load_data, down_sample_data
from src.data_generation import train_path, test_path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def pre_process_data(X, y):
    y = np.minimum(y, 1)  # convert all values to 0 or 1
    y = y.ravel()  # flatten y_train
    X += 1  # make all values positive
    return X, y
    
    


def run():
    # load data
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)

    # pre-process data (make all X values positive, 
    # convert all y values to 0 or 1)
    X_train, y_train = pre_process_data(X_train, y_train)
    X_test, y_test = pre_process_data(X_test, y_test)

    # down sample data so that the number of healthy 
    # boids is equal to the number of faulty boids
    X_train, y_train = down_sample_data(X_train, y_train)
    X_test, y_test = down_sample_data(X_test, y_test)

    # get model and predict
    model = nb.GaussianNB().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate accuracy and confusion matrix for the test set
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification = classification_report(y_test, y_pred)

    print("Naive Bayes")
    print()
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion)
    print()
    print("Classification Report:")
    print(classification)

    return model
