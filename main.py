# === Import Libraries ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === Step 1: Load Data ===
csv_file = "MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
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
from scipy import stats
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

# Save final features list to a file
with open("models/final_features.txt", "w") as f:
    for feature in X.columns:
        f.write(feature + "\n")

print("Final feature list saved to 'models/final_features.txt'")

# === Step 7: Temporal Split (More Realistic) ===
# Sort by index assuming temporal order, or add timestamp-based sorting if available
print("\n=== Using Temporal Split ===")
split_idx = int(len(df) * 0.7)  # 70% for training
X_train_temp = X.iloc[:split_idx]
X_test_temp = X.iloc[split_idx:]
y_train_temp = y.iloc[:split_idx]
y_test_temp = y.iloc[split_idx:]

print(f"Temporal split - Train: {len(X_train_temp)}, Test: {len(X_test_temp)}")
print(f"Train class distribution: {y_train_temp.value_counts().to_dict()}")
print(f"Test class distribution: {y_test_temp.value_counts().to_dict()}")