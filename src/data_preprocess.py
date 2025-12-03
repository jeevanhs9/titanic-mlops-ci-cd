import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_raw_data(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)

    # Keep only useful columns for simplicity
    cols = [
        "Survived", "Pclass", "Sex", "Age",
        "SibSp", "Parch", "Fare", "Embarked"
    ]
    df = df[cols]

    # Drop rows where target is missing (just in case)
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing transformer for numeric + categorical features."""
    numeric_features = ["Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Pclass", "Sex", "Embarked"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor
