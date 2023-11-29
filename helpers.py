from sklearn.model_selection import cross_val_score, KFold

def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test, num_folds=5):
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
