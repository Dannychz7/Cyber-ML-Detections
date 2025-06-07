import pandas as pd
import joblib
import sys
import torch
import torch.nn as nn
import numpy as np

# Define the same neural network architecture used in training
class DDosDetectionNet(nn.Module):
    def __init__(self, input_dim):
        super(DDosDetectionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def main(input_csv):
    try:
        # Load model architecture info
        model_info = joblib.load("models/ddos_model_info.pkl")
        input_dim = model_info['input_dim']
        expected_features = model_info['features']
        
        print(f"Loaded model info - Input dim: {input_dim}")
        print(f"Expected features ({len(expected_features)}): {expected_features}")
        
        # Initialize model with correct architecture
        model = DDosDetectionNet(input_dim)
        
        # Load trained model weights
        model.load_state_dict(torch.load("models/ddos_torch_model.pt"))
        model.eval()  # Set to evaluation mode
        
        print("✅ PyTorch model loaded successfully")
        
    except FileNotFoundError as e:
        print(f"ERROR: Model files not found. Make sure you've trained the model first.")
        print(f"Missing file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    # Load input data
    try:
        df = pd.read_csv(input_csv, on_bad_lines='warn')
        print(f"✅ Loaded input data: {df.shape}")
        print(f"Input data columns: {list(df.columns)}")
        
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        sys.exit(1)

    # Check for expected features
    missing_features = [feat for feat in expected_features if feat not in df.columns]
    if missing_features:
        print(f"ERROR: Missing expected features in input data: {missing_features}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Select features in correct order
    X = df[expected_features]
    print(f"✅ Selected {X.shape[1]} features for prediction")

    # Handle missing values and infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().any().any():
        print("⚠️  Warning: Found missing/infinite values, filling with 0")
        X = X.fillna(0)

    try:
        # Load and apply scaler
        scaler = joblib.load("models/ddos_scaler.pkl")
        X_scaled = scaler.transform(X)
        print("✅ Applied feature scaling")
        
    except FileNotFoundError:
        print("WARNING: Scaler not found, using unscaled features")
        X_scaled = X.values
    except Exception as e:
        print(f"ERROR applying scaler: {e}")
        sys.exit(1)

    # Convert to PyTorch tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Make predictions
    try:
        with torch.no_grad():
            # Get probability predictions
            y_pred_proba = model(X_tensor).squeeze().numpy()
            
            # Convert probabilities to binary predictions (threshold = 0.5)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Convert to more interpretable labels
            y_pred_labels = ['ATTACK' if pred == 1 else 'BENIGN' for pred in y_pred]
        
        print(f"✅ Generated predictions for {len(y_pred)} samples")
        
        # Print prediction summary
        attack_count = np.sum(y_pred)
        benign_count = len(y_pred) - attack_count
        print(f"📊 Prediction Summary:")
        print(f"   - BENIGN: {benign_count} ({benign_count/len(y_pred)*100:.1f}%)")
        print(f"   - ATTACK: {attack_count} ({attack_count/len(y_pred)*100:.1f}%)")
        
        # Show confidence distribution
        high_confidence = np.sum((y_pred_proba > 0.8) | (y_pred_proba < 0.2))
        print(f"   - High confidence predictions: {high_confidence}/{len(y_pred)} ({high_confidence/len(y_pred)*100:.1f}%)")
        
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        sys.exit(1)

    # Add predictions to dataframe
    df['predicted_label'] = y_pred_labels
    df['predicted_binary'] = y_pred
    df['predicted_prob'] = y_pred_proba
    df['confidence'] = np.maximum(y_pred_proba, 1 - y_pred_proba)  # Distance from 0.5

    # Save predictions to CSV
    output_csv = input_csv.replace(".csv", "_predictions.csv")
    try:
        df.to_csv(output_csv, index=False)
        print(f"✅ Predictions saved to: {output_csv}")
        
        # Save summary statistics
        summary_csv = input_csv.replace(".csv", "_prediction_summary.csv")
        summary_df = pd.DataFrame({
            'Metric': ['Total Samples', 'Benign Predictions', 'Attack Predictions', 
                      'Avg Confidence', 'High Confidence Count'],
            'Value': [len(y_pred), benign_count, attack_count, 
                     np.mean(df['confidence']), high_confidence]
        })
        summary_df.to_csv(summary_csv, index=False)
        print(f"✅ Summary saved to: {summary_csv}")
        
    except Exception as e:
        print(f"ERROR saving results: {e}")
        sys.exit(1)

    # Show sample predictions
    print(f"\n📋 Sample Predictions (first 10 rows):")
    sample_cols = ['predicted_label', 'predicted_prob', 'confidence']
    if 'Label' in df.columns:  # If ground truth is available
        sample_cols = ['Label'] + sample_cols
    print(df[sample_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_ddos.py <input_csv_file>")
        print("\nExample:")
        print("  python predict_ddos.py test_data.csv")
        print("\nRequired files:")
        print("  - models/ddos_torch_model.pt (trained PyTorch model)")
        print("  - models/ddos_model_info.pkl (model architecture info)")
        print("  - models/ddos_scaler.pkl (feature scaler)")
        sys.exit(1)

    input_csv = sys.argv[1]
    print(f"🚀 Starting DDoS prediction on: {input_csv}")
    main(input_csv)