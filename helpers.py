from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate_data(features, output, model):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        output,
        test_size=0.25,  # Adjust the test_size as needed
        random_state=100  # You can set a random seed for reproducibility
    )

    # Train the provided model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, num_folds=5):
    # Create a cross-validation object (KFold or StratifiedKFold)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf)

    # Print the cross-validation scores
    print(f"{model_name} Cross-Validation Scores:", cv_scores)

    # Print the mean and standard deviation of the scores
    print(f"{model_name} Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"{model_name} Standard Deviation: {cv_scores.std():.4f}")

    # Train the model on the entire training set
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    test_accuracy = model.score(X_test, y_test)
    print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
