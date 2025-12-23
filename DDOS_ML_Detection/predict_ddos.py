import pandas as pd
import joblib
import sys

def main(input_csv):
    # Load saved model
    model = joblib.load("models/ddos_xgb_model.pkl")

    # Load expected features list
    with open("models/final_features.txt", "r") as f:
        expected_features = [line.strip() for line in f.readlines()]

    print(f"Loaded expected features ({len(expected_features)}): {expected_features}")

    try:
        df = pd.read_csv(input_csv, on_bad_lines='warn')
        print(df.head())
    except Exception as e:
        print(f"Error reading CSV: {e}")
    print(f"Input data columns: {list(df.columns)}")

    # Ensure all expected features are present
    missing_features = [feat for feat in expected_features if feat not in df.columns]
    if missing_features:
        print(f"ERROR: Missing expected features in input data: {missing_features}")
        sys.exit(1)

    # Select features in correct order
    X = df[expected_features]

# 1. Load and apply scaler 
    # (The scaler MUST have been trained on the same 7 or 12 features)
    scaler = joblib.load("models/ddos_scaler_xgb.pkl")
    X_scaled = scaler.transform(X)

    # 2. Predict using the SCALED data, not the raw 'X'
    y_pred = model.predict(X_scaled) 
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    # Output predictions
    df['predicted_label'] = y_pred
    df['predicted_prob'] = y_pred_proba

    # Save predictions to CSV
    output_csv = input_csv.replace(".csv", "_predictions.csv")
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_ddos.py <input_csv_file>")
        sys.exit(1)

    input_csv = sys.argv[1]
    main(input_csv)
