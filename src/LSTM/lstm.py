from src.data_pre_processing import load_timeseries_data,down_sample_data,data_pre_processing
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout,Bidirectional
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,matthews_corrcoef
import pickle
from src.data_generation import train_path, test_path


def lstm_method():
    #  uploading data
    X_train, y_train = load_timeseries_data(train_path)
    X_test, y_test = load_timeseries_data(test_path)
    y_train[y_train == 2] = 1
    y_train[y_train == 3] = 1
    y_test[y_test == 2] = 1
    y_test[y_test == 3] = 1
    X_train, y_train = down_sample_data (X_train, y_train)


    model = Sequential()
    model.add(LSTM(100, activation='relu',input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))  # Adjust the dropout rate as needed

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=100)#, class_weight=class_weight


    # Make predictions on the test set, not the training set
    y_pred = model.predict(X_test)

    # Round the predictions to 0 or 1 (assuming a threshold of 0.5)
    y_pred = np.round(y_pred)

    # Calculate accuracy and confusion matrix for the test set
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    # save the iris classification model as a pickle file
    model_pkl_file = "lstm_model100.pkl"

    with open(model_pkl_file, 'wb') as file:
        pickle.dump(model, file)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("MCC:")
    print(matthews_corrcoef(y_test, y_pred))

if __name__ == "__main__":
    lstm_method()


