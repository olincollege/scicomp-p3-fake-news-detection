""" helpers.py
This file defines functions to help evaluate different machine learning models and the effect of
different sets of features. The train_and_evaluate_data function makes it easy to test multiple
sets of features. The train_and_evaluate_models function helps with testing multiple models quickly
with built-in cross validation.
"""
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate_data(features, output, model):
    """
    Train and evaluate a machine learning model using the provided features and output. Primarily
    used for evaluating different sets of features.

    Args:
        features: An array of the input features for training and testing the model.
        output: An array of the corresponding output or target variable.
        model: An instance of a scikit-learn compatible machine learning model.

    Returns:
        N/A
        The function prints the evaluation metrics, including accuracy, precision, recall, and F1
        score.
    """
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        output,
        # The test_size parameter determines the proportion used for testing (here, 25%)
        test_size=0.25,
        # The random_state parameter ensures reproducibility by fixing the random seed for the split
        random_state=10000
    )

    # Train the provided model
    model.fit(x_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1_value = f1_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_value}")

def train_and_evaluate_model(model, model_name, x_train, y_train, x_test, y_test, num_folds=5):
    """
    Train and evaluate a machine learning model using cross-validation and a separate test set.

    Args:
        model: An instance of a scikit-learn compatible machine learning model.
        model_name: A string representing the name of the model for display purposes.
        x_train: An array of the input features for training the model.
        y_train: An array of the corresponding output or target variable for training.
        x_test: An array of the input features for testing the model.
        y_test: An array of the corresponding output or target variable for testing.
        num_folds: An integer representing the number of folds for cross-validation. Default is 5.

    Returns:
        None: The function prints cross-validation scores, mean accuracy, standard deviation, and
        test accuracy.
    """
    # Create a cross-validation object
    k_fold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(model, x_train, y_train, cv=k_fold)

    # Print the cross-validation scores
    print(f"{model_name} Cross-Validation Scores:", cv_scores)

    # Print the mean and standard deviation of the scores
    print(f"{model_name} Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"{model_name} Standard Deviation: {cv_scores.std():.4f}")

    # Train the model on the entire training set
    model.fit(x_train, y_train)

    # Evaluate the model on the test set
    test_accuracy = model.score(x_test, y_test)
    print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
