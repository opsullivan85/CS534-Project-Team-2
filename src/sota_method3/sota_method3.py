from src.data_pre_processing import load_timeseries_data, data_pre_processing
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout,Bidirectional
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def lstm_method():
    X, y = data_pre_processing.load_timeseries_data()
    # converting labels 2 and 3 as 1
    y[y == 2] = 1
    y[y == 3] = 1
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)#shuffle=False


    # Define the LSTM model

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))  # Adjust the dropout rate as needed
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    #class_weight = {0: 1, 1: 10, 2: 10, 3: 10}  # Adjust the weights as needed
    model.fit(X_train, y_train, epochs=10, batch_size=16)#, class_weight=class_weight


    # Make predictions on the test set, not the training set
    y_pred = model.predict(X_test)

    # Round the predictions to 0 or 1 (assuming a threshold of 0.5)
    y_pred = np.round(y_pred)

    # Calculate accuracy and confusion matrix for the test set
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    lstm_method()
