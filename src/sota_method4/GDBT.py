from src.data_pre_processing import down_sample_data, load_data
from src.data_generation import train_path, test_path
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# tdqm progres bar - to keep tabs
def gradientB_decision_tree():
    # Load your dataset and split it into features (X) and labels (y)
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)
    # Convert 2D to 1D
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    # print(y_train.shape)

    # Combine labels 2 and 3 into a single class
    y_train[y_train == 2] = 1
    y_train[y_train == 3] = 1
    y_test[y_test == 2] = 1
    y_test[y_test == 3] = 1

    X_train, y_train = down_sample_data(X_train, y_train)

    # Create and configure the Gradient Boosted Decision Tree classifier
    clf = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=10, random_state=42, verbose=2)

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    model_filename = 'gradient_boosted_model.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(clf, model_file)
    print(f"Trained model saved to {model_filename}")

    # Make predictions on the test set
    y_test_pred = clf.predict(X_test)

    # Evaluate the performance of the classifier on the test data.
    accuracy_test = np.mean(y_test_pred == y_test)
    confusion_test = confusion_matrix(y_test, y_test_pred)


    # Print the accuracy and results for the test set.
    print('Accuracy on Test Set:', accuracy_test)
    print("Confusion Matrix for Test Set:")
    print(confusion_test)
    print("Classification Report for Test Set:")
    print(classification_report(y_test, y_test_pred))
    print(matthews_corrcoef(y_test, y_test_pred))
    
    return clf
