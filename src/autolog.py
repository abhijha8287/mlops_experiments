import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

# Initialize DAGsHub (NO create_repo)
dagshub.init(
    repo_owner="abhijha8287",
    repo_name="mlops_experiments",
    mlflow=True
)

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/abhijha8287/mlops_experiments.mlflow")

# Load data
wine = load_wine()
x = wine.data
y = wine.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Model params
max_depth = 90
n_estimators = 70

mlflow.set_experiment("mlops_experiments")
mlflow.autolog()

with mlflow.start_run():
    clf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )
    
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Log parameters & metrics
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)

    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=wine.target_names,
                yticklabels=wine.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log model
    mlflow.sklearn.log_model(clf, "random_forest_model")

    # Tags
    mlflow.set_tags({
        "author": "abhijha",
        "model": "RandomForestClassifier"
    })

    print("Classification Report:\n", report)
