import os
import glob
import datetime
import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.data_preprocess import load_raw_data, build_preprocessor


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_new_model_path(model_dir: str) -> str:
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(model_dir, f"model_{timestamp}.pkl")


def update_latest_symlink(model_path: str, latest_path: str):
    # Just copy the file path logically – here we really overwrite latest_model.pkl
    import shutil
    shutil.copy2(model_path, latest_path)


def main():
    config = load_config()

    data_path = config["data"]["path"]
    target_col = config["data"]["target"]
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]

    model_dir = config["paths"]["model_dir"]
    latest_model_path = config["paths"]["latest_model"]

    # Load data
    X, y = load_raw_data(data_path, target_col)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Build preprocessing
    preprocessor = build_preprocessor(X)

    # Build model
    rf_params = config["model"]["params"]
    model = RandomForestClassifier(**rf_params)

    # Create full pipeline: preprocess + model
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # Train
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # Save model with version
    new_model_path = get_new_model_path(model_dir)
    joblib.dump(clf, new_model_path)
    print(f"Saved model version: {new_model_path}")

    # Update latest_model.pkl
    update_latest_symlink(new_model_path, latest_model_path)
    print(f"Updated latest model at: {latest_model_path}")


if __name__ == "__main__":
    main()
