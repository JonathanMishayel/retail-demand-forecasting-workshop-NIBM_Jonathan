from src.utils.io_utils import load_csv
from src.config.config import DATA_FEATURE_PATH, FEATURE_COLS, TARGET_COL, MODEL_PATH
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# src/train_model.py


# train_model.py
def train_model(X_train, y_train, model_path=None):
    from src.config.config import MODEL_PATH
    import os, joblib
    from sklearn.ensemble import RandomForestRegressor

    if model_path is None:
        model_path = MODEL_PATH

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model
