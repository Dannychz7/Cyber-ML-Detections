# === Import Libraries ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from config import DATAPATH_FRIDAY_WORK_HRS_AFTERNOON

# === Step 1: Load Data ===
csv_file = DATAPATH_FRIDAY_WORK_HRS_AFTERNOON
df = pd.read_csv(csv_file)

# === Step 2: Clean Data ===
df.columns = df.columns.str.strip()  # remove leading/trailing spaces
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print(f"Dataset shape after cleaning: {df.shape}")

# === Step 3: Binary Labeling ===
df['binary_label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# === Step 4: Remove More Aggressive Leaky Features ===
df = df.drop(columns=['Label'])

# Expanded list of potentially leaky features
leaky_features = [
    'Flow Bytes/s', 'Flow Packets/s', 'Init_Win_bytes_backward', 'Init_Win_bytes_forward',
    'Fwd Header Length', 'Bwd Header Length',
    # Additional potentially leaky features
    'Flow Duration',  # Attack duration patterns
    'Total Fwd Packets', 'Total Backward Packets',  # Too distinctive for attacks
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',  # Size patterns
    'Fwd Packets/s', 'Bwd Packets/s',  # Rate-based features are very leaky
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',  # Timing signatures
    'Fwd IAT Total', 'Bwd IAT Total',  # Timing aggregates
    'Down/Up Ratio',  # Often perfect separator
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size'  # Size signatures
]

df = df.drop(columns=[col for col in leaky_features if col in df.columns])
print(f"Dropped {len([col for col in leaky_features if col in df.columns])} potentially leaky features")

# Drop constant columns
initial_cols = len(df.columns)
df = df.loc[:, df.nunique() > 1]
print(f"Dropped {initial_cols - len(df.columns)} constant columns")

# === Step 5: Advanced Leakage Detection ===
print("\n=== Advanced Leakage Detection ===")
X_temp = df.drop(columns=['binary_label'])
y_temp = df['binary_label']

# Check for perfect separators
perfect_separators = []
near_perfect_separators = []

for col in X_temp.columns:
    benign_vals = set(X_temp[y_temp == 0][col].unique())
    attack_vals = set(X_temp[y_temp == 1][col].unique())
    overlap = benign_vals & attack_vals
    
    # Perfect separation
    if len(overlap) == 0 and len(benign_vals) > 1 and len(attack_vals) > 1:
        perfect_separators.append(col)
        print(f"Perfect separator: {col}")
    
    # Near-perfect separation (< 5% overlap)
    elif len(overlap) > 0:
        total_unique = len(benign_vals | attack_vals)
        overlap_ratio = len(overlap) / total_unique
        if overlap_ratio < 0.05:
            near_perfect_separators.append((col, overlap_ratio))
            print(f"Near-perfect separator: {col} (overlap: {overlap_ratio:.3f})")

# Check statistical separation
print("\n=== Statistical Separation Analysis ===")
high_separation_features = []
for col in X_temp.columns:
    benign_data = X_temp[y_temp == 0][col]
    attack_data = X_temp[y_temp == 1][col]
    
    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(benign_data, attack_data, alternative='two-sided')
    
    if p_value < 1e-50:  # Extremely low p-value
        effect_size = 1 - (2 * statistic) / (len(benign_data) * len(attack_data))
        high_separation_features.append((col, p_value, abs(effect_size)))
        print(f"High separation: {col} (p={p_value:.2e}, effect_size={abs(effect_size):.3f})")

# Remove problematic features
problematic_features = perfect_separators + [feat[0] for feat in near_perfect_separators]
if problematic_features:
    df = df.drop(columns=problematic_features)
    print(f"\nRemoved {len(problematic_features)} highly separating features: {problematic_features}")

# Show distribution examples for top separating features
print("\n=== Distribution Analysis of Top Features ===")
remaining_features = [col for col in X_temp.columns if col not in problematic_features]
for col in remaining_features[:3]:  # Show top 3 remaining features
    benign_data = X_temp[y_temp == 0][col]
    attack_data = X_temp[y_temp == 1][col]
    print(f"\n{col}:")
    print(f"  Benign - Mean: {benign_data.mean():.3f}, Std: {benign_data.std():.3f}")
    print(f"  Attack - Mean: {attack_data.mean():.3f}, Std: {attack_data.std():.3f}")
    print(f"  Separation: {abs(benign_data.mean() - attack_data.mean()) / (benign_data.std() + attack_data.std()):.3f}")

# === Step 6: Features/Target Separation ===
X = df.drop(columns=['binary_label'])
y = df['binary_label']
X.columns = [str(col) for col in X.columns]

print(f"Final feature count: {len(X.columns)}")
print(f"Remaining features: {list(X.columns)}")

# Create models directory if it doesn't exist
import os
os.makedirs("models", exist_ok=True)

# Save final features list to a file
with open("models/final_features.txt", "w") as f:
    for feature in X.columns:
        f.write(feature + "\n")

print("Final feature list saved to 'models/final_features.txt'")

# === Step 7: Temporal Split (More Realistic) ===
print("\n=== Using Temporal Split ===")
split_idx = int(len(df) * 0.7)  # 70% for training
X_train_temp = X.iloc[:split_idx]
X_test_temp = X.iloc[split_idx:]
y_train_temp = y.iloc[:split_idx]
y_test_temp = y.iloc[split_idx:]

print(f"Temporal split - Train: {len(X_train_temp)}, Test: {len(X_test_temp)}")
print(f"Train class distribution: {y_train_temp.value_counts().to_dict()}")
print(f"Test class distribution: {y_test_temp.value_counts().to_dict()}")

# === Step 8: Feature Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_temp)
X_test_scaled = scaler.transform(X_test_temp)

# Convert scaled NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_temp.values, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_temp.values, dtype=torch.float32)

# Wrap tensors in DataLoaders
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# === Step 9: Neural Network Model Definition ===
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

# Initialize model, loss, and optimizer
input_dim = X_train_scaled.shape[1]
model = DDosDetectionNet(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# === Step 10: Training Loop ===
print("\n=== Training Neural Network ===")
epochs = 50
train_losses = []
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# === Step 11: Cross-Validation with PyTorch ===
print("\n=== Cross-Validation Results ===")

def pytorch_cv_score(X, y, n_splits=5):
    """Custom cross-validation for PyTorch model"""
    cv_scores = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X, y):
        # Split data
        X_train_cv = X[train_idx]
        X_val_cv = X[val_idx]
        y_train_cv = y[train_idx]
        y_val_cv = y[val_idx]
        
        # Scale features
        scaler_cv = StandardScaler()
        X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
        X_val_cv_scaled = scaler_cv.transform(X_val_cv)
        
        # Convert to tensors
        X_train_cv_tensor = torch.tensor(X_train_cv_scaled, dtype=torch.float32)
        y_train_cv_tensor = torch.tensor(y_train_cv, dtype=torch.float32)
        X_val_cv_tensor = torch.tensor(X_val_cv_scaled, dtype=torch.float32)
        
        # Create model
        model_cv = DDosDetectionNet(X_train_cv_scaled.shape[1])
        criterion_cv = nn.BCELoss()
        optimizer_cv = optim.Adam(model_cv.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Train
        model_cv.train()
        for epoch in range(20):  # Fewer epochs for CV
            optimizer_cv.zero_grad()
            outputs = model_cv(X_train_cv_tensor).squeeze()
            loss = criterion_cv(outputs, y_train_cv_tensor)
            loss.backward()
            optimizer_cv.step()
        
        # Evaluate
        model_cv.eval()
        with torch.no_grad():
            val_pred_probs = model_cv(X_val_cv_tensor).squeeze().numpy()
            cv_score = roc_auc_score(y_val_cv, val_pred_probs)
            cv_scores.append(cv_score)
    
    return np.array(cv_scores)

cv_scores = pytorch_cv_score(X_train_scaled, y_train_temp.values)
print(f"CV ROC-AUC scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# === Step 12: Evaluate on Test Set ===
print("\n=== Test Set Results ===")
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_tensor).squeeze().numpy()
    y_preds = (y_pred_probs >= 0.5).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test_temp, y_preds))
print("\nClassification Report:")
print(classification_report(y_test_temp, y_preds))
print(f"ROC-AUC Score: {roc_auc_score(y_test_temp, y_pred_probs):.4f}")

# === Step 13: Random Split Comparison ===
print("\n=== Comparison: Random Split Results ===")
X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train_rand_scaled = scaler.fit_transform(X_train_rand)
X_test_rand_scaled = scaler.transform(X_test_rand)

# Convert to tensors for random split
X_train_rand_tensor = torch.tensor(X_train_rand_scaled, dtype=torch.float32)
y_train_rand_tensor = torch.tensor(y_train_rand.values, dtype=torch.float32)
X_test_rand_tensor = torch.tensor(X_test_rand_scaled, dtype=torch.float32)

# Train model on random split
model_rand = DDosDetectionNet(X_train_rand_scaled.shape[1])
criterion_rand = nn.BCELoss()
optimizer_rand = optim.Adam(model_rand.parameters(), lr=0.001, weight_decay=1e-5)

model_rand.train()
for epoch in range(30):
    optimizer_rand.zero_grad()
    outputs = model_rand(X_train_rand_tensor).squeeze()
    loss = criterion_rand(outputs, y_train_rand_tensor)
    loss.backward()
    optimizer_rand.step()

# Evaluate random split model
model_rand.eval()
with torch.no_grad():
    y_pred_rand_probs = model_rand(X_test_rand_tensor).squeeze().numpy()
    y_pred_rand = (y_pred_rand_probs >= 0.5).astype(int)

print("Random Split Confusion Matrix:")
print(confusion_matrix(y_test_rand, y_pred_rand))
print(f"Random Split ROC-AUC: {roc_auc_score(y_test_rand, y_pred_rand_probs):.4f}")

# === Step 14: Class Distribution Analysis ===
print("\n=== Class Distribution Analysis ===")
print("Overall class distribution:")
print(y.value_counts())
print(f"Class balance ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}")

# === Step 15: Deep Dataset Analysis ===
print("\n=== Deep Dataset Analysis ===")

# Check if we have a realistic dataset left
if len(X.columns) < 5:
    print("WARNING: Too few features remaining. Dataset may be fundamentally flawed.")

# Check for constant attack values (another type of leakage)
print(f"\n=== Checking for Constant Attack Values ===")
constant_attack_features = []
for col in X.columns:
    attack_vals = X_temp[y_temp == 1][col].unique()
    if len(attack_vals) == 1:
        constant_attack_features.append(col)
        print(f"Constant attack values: {col} = {attack_vals[0]}")

# Remove features with constant attack values and retrain if needed
if constant_attack_features:
    df_clean = df.drop(columns=constant_attack_features)
    print(f"Removed {len(constant_attack_features)} constant-attack-value features")
    
    # Retrain with cleaner features
    X_clean = df_clean.drop(columns=['binary_label'])
    y_clean = df_clean['binary_label']
    
    print(f"Final clean feature count: {len(X_clean.columns)}")
    print(f"Clean features: {list(X_clean.columns)}")
    
    # Re-run the model with truly clean features
    split_idx = int(len(df_clean) * 0.7)
    X_train_clean = X_clean.iloc[:split_idx]
    X_test_clean = X_clean.iloc[split_idx:]
    y_train_clean = y_clean.iloc[:split_idx]
    y_test_clean = y_clean.iloc[split_idx:]
    
    scaler_clean = StandardScaler()
    X_train_clean_scaled = scaler_clean.fit_transform(X_train_clean)
    X_test_clean_scaled = scaler_clean.transform(X_test_clean)
    
    # Convert to tensors
    X_train_clean_tensor = torch.tensor(X_train_clean_scaled, dtype=torch.float32)
    y_train_clean_tensor = torch.tensor(y_train_clean.values, dtype=torch.float32)
    X_test_clean_tensor = torch.tensor(X_test_clean_scaled, dtype=torch.float32)
    
    # Train clean model
    model_clean = DDosDetectionNet(X_train_clean_scaled.shape[1])
    criterion_clean = nn.BCELoss()
    optimizer_clean = optim.Adam(model_clean.parameters(), lr=0.001, weight_decay=1e-5)
    
    model_clean.train()
    for epoch in range(40):
        optimizer_clean.zero_grad()
        outputs = model_clean(X_train_clean_tensor).squeeze()
        loss = criterion_clean(outputs, y_train_clean_tensor)
        loss.backward()
        optimizer_clean.step()
    
    # Evaluate clean model
    model_clean.eval()
    with torch.no_grad():
        y_pred_clean_probs = model_clean(X_test_clean_tensor).squeeze().numpy()
        y_pred_clean = (y_pred_clean_probs >= 0.5).astype(int)
    
    print(f"\n=== FINAL CLEAN MODEL RESULTS ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_clean, y_pred_clean))
    print("\nClassification Report:")
    print(classification_report(y_test_clean, y_pred_clean))
    print(f"ROC-AUC Score: {roc_auc_score(y_test_clean, y_pred_clean_probs):.4f}")

# === Step 16: Save Trained Pipeline ===
joblib.dump(scaler, "models/ddos_scaler.pkl")
torch.save(model.state_dict(), "models/ddos_torch_model.pt")

# Also save the model architecture info
model_info = {
    'input_dim': input_dim,
    'features': list(X.columns)
}
joblib.dump(model_info, "models/ddos_model_info.pkl")

print("✅ Saved PyTorch model and scaler to disk.")

# Final reality check
tn, fp, fn, tp = confusion_matrix(y_test_temp, y_preds).ravel()
print(f"\n=== Final Reality Check ===")
print(f"If this were a real-world deployment:")
print(f"- False Positive Rate: {fp/(fp+tn):.4f} ({fp} benign flows flagged as attacks)")
print(f"- False Negative Rate: {fn/(fn+tp):.4f} ({fn} attacks missed)")
if fn > 0:
    print(f"- This means missing 1 in {(tp+fn)//fn if fn > 0 else 'N/A'} DDoS attacks")
if fp > 0:
    print(f"- And falsely alerting on 1 in {(tn+fp)//fp if fp > 0 else 'N/A'} benign flows")

# Suggest next steps
print(f"\n=== Final Recommendations ===")
print("✅ SUCCESS: You now have a PyTorch-based DDoS detection model!")
print("📊 Neural network performance with meaningful trade-offs")
print("🎯 Next steps for improvement:")
print("1. Tune the decision threshold to balance precision/recall")
print("2. Try different neural network architectures (deeper/wider networks)")
print("3. Implement early stopping and learning rate scheduling")
print("4. Add domain knowledge features (burst patterns, flow duration bins)")
print("5. Test on different DDoS attack types")
print("6. Consider ensemble methods combining multiple neural networks")
print("7. Experiment with other PyTorch optimizers (SGD, RMSprop)")
print("8. Add regularization techniques (batch normalization, different dropout rates)")