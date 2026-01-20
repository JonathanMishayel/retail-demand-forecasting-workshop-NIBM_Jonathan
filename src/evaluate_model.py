from src.utils.io_utils import load_csv
from src.utils.metrics import calculate_mape
from src.config.config import DATA_FEATURE_PATH, FEATURE_COLS, TARGET_COL, MODEL_PATH
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# src/evaluate.py

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and print metrics.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return {"MAE": mae, "MSE": mse, "R2": r2}

