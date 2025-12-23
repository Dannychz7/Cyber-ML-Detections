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
from datetime import datetime
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import base64

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from config import DATAPATH_FRIDAY_WORK_HRS_AFTERNOON

# === Output Setup ===
# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Initialize output files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"results/ddos_analysis_log_{timestamp}.txt"
report_file = f"results/ddos_analysis_report_{timestamp}.pdf"

class OutputLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.content = []
    
    def log(self, message):
        self.content.append(str(message))
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(str(message) + '\n')
    
    def get_content(self):
        return '\n'.join(self.content)

logger = OutputLogger(log_file)

# === Step 1: Load Data ===
logger.log("=== DDoS Detection Model Analysis ===")
logger.log(f"Analysis started at: {datetime.now()}")
logger.log("")

csv_file = DATAPATH_FRIDAY_WORK_HRS_AFTERNOON
df = pd.read_csv(csv_file)

logger.log("=== Step 1: Data Loading ===")
logger.log(f"Loaded dataset from: {csv_file}")
logger.log(f"Initial dataset shape: {df.shape}")

# === Step 2: Clean Data ===
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan)
initial_rows = len(df)
df = df.dropna()
cleaned_rows = len(df)

logger.log("")
logger.log("=== Step 2: Data Cleaning ===")
logger.log(f"Rows before cleaning: {initial_rows}")
logger.log(f"Rows after cleaning: {cleaned_rows}")
logger.log(f"Rows removed: {initial_rows - cleaned_rows}")
logger.log(f"Dataset shape after cleaning: {df.shape}")

# === Step 3: Binary Labeling ===
df['binary_label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
label_counts = df['binary_label'].value_counts()

logger.log("")
logger.log("=== Step 3: Binary Labeling ===")
logger.log(f"Benign samples: {label_counts[0]}")
logger.log(f"Attack samples: {label_counts[1]}")
logger.log(f"Class balance ratio (Benign/Attack): {label_counts[0] / label_counts[1]:.2f}")

# === Step 4: Remove Aggressive Leaky Features ===
df = df.drop(columns=['Label'])

leaky_features = [
    'Flow Bytes/s', 'Flow Packets/s', 'Init_Win_bytes_backward', 'Init_Win_bytes_forward',
    'Fwd Header Length', 'Bwd Header Length',
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packets/s', 'Bwd Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Bwd IAT Total',
    'Down/Up Ratio',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
]

dropped_leaky = [col for col in leaky_features if col in df.columns]
df = df.drop(columns=dropped_leaky)

logger.log("")
logger.log("=== Step 4: Remove Leaky Features ===")
logger.log(f"Potentially leaky features identified: {len(leaky_features)}")
logger.log(f"Actually dropped: {len(dropped_leaky)}")
logger.log("Dropped features:")
for feature in dropped_leaky:
    logger.log(f"  - {feature}")

# Drop constant columns
initial_cols = len(df.columns)
df = df.loc[:, df.nunique() > 1]
constant_cols_dropped = initial_cols - len(df.columns)

logger.log("")
logger.log(f"Constant columns dropped: {constant_cols_dropped}")

# === Step 5: Advanced Leakage Detection ===
logger.log("")
logger.log("=== Step 5: Advanced Leakage Detection ===")
X_temp = df.drop(columns=['binary_label'])
y_temp = df['binary_label']

perfect_separators = []
near_perfect_separators = []

for col in X_temp.columns:
    benign_vals = set(X_temp[y_temp == 0][col].unique())
    attack_vals = set(X_temp[y_temp == 1][col].unique())
    overlap = benign_vals & attack_vals
    
    if len(overlap) == 0 and len(benign_vals) > 1 and len(attack_vals) > 1:
        perfect_separators.append(col)
        logger.log(f"Perfect separator found: {col}")
    elif len(overlap) > 0:
        total_unique = len(benign_vals | attack_vals)
        overlap_ratio = len(overlap) / total_unique
        if overlap_ratio < 0.05:
            near_perfect_separators.append((col, overlap_ratio))
            logger.log(f"Near-perfect separator: {col} (overlap: {overlap_ratio:.3f})")

# Statistical separation analysis
logger.log("")
logger.log("Statistical Separation Analysis:")
high_separation_features = []
for col in X_temp.columns:
    benign_data = X_temp[y_temp == 0][col]
    attack_data = X_temp[y_temp == 1][col]
    
    statistic, p_value = stats.mannwhitneyu(benign_data, attack_data, alternative='two-sided')
    
    if p_value < 1e-50:
        effect_size = 1 - (2 * statistic) / (len(benign_data) * len(attack_data))
        high_separation_features.append((col, p_value, abs(effect_size)))
        logger.log(f"High separation: {col} (p={p_value:.2e}, effect_size={abs(effect_size):.3f})")

# Remove problematic features
problematic_features = perfect_separators + [feat[0] for feat in near_perfect_separators]
if problematic_features:
    df = df.drop(columns=problematic_features)
    logger.log(f"\nRemoved {len(problematic_features)} highly separating features:")
    for feat in problematic_features:
        logger.log(f"  - {feat}")

# Distribution analysis
logger.log("")
logger.log("Distribution Analysis of Top Features:")
remaining_features = [col for col in X_temp.columns if col not in problematic_features]
for col in remaining_features[:3]:
    benign_data = X_temp[y_temp == 0][col]
    attack_data = X_temp[y_temp == 1][col]
    separation_score = abs(benign_data.mean() - attack_data.mean()) / (benign_data.std() + attack_data.std())
    
    logger.log(f"\n{col}:")
    logger.log(f"  Benign - Mean: {benign_data.mean():.3f}, Std: {benign_data.std():.3f}")
    logger.log(f"  Attack - Mean: {attack_data.mean():.3f}, Std: {attack_data.std():.3f}")
    logger.log(f"  Separation Score: {separation_score:.3f}")

# === Step 6: Features/Target Separation ===
X = df.drop(columns=['binary_label'])
y = df['binary_label']
X.columns = [str(col) for col in X.columns]

logger.log("")
logger.log("=== Step 6: Final Feature Set ===")
logger.log(f"Final feature count: {len(X.columns)}")
logger.log("Remaining features:")
for i, feature in enumerate(X.columns, 1):
    logger.log(f"  {i}. {feature}")

# Save final features list
with open("models/final_features.txt", "w") as f:
    for feature in X.columns:
        f.write(feature + "\n")

logger.log(f"\nFinal feature list saved to 'models/final_features.txt'")

# === Step 7: Temporal Split ===
logger.log("")
logger.log("=== Step 7: Temporal Split ===")
split_idx = int(len(df) * 0.7)
X_train_temp = X.iloc[:split_idx]
X_test_temp = X.iloc[split_idx:]
y_train_temp = y.iloc[:split_idx]
y_test_temp = y.iloc[split_idx:]

train_class_dist = y_train_temp.value_counts()
test_class_dist = y_test_temp.value_counts()

logger.log(f"Temporal split - Train: {len(X_train_temp)}, Test: {len(X_test_temp)}")
logger.log(f"Train class distribution: Benign={train_class_dist[0]}, Attack={train_class_dist[1]}")
logger.log(f"Test class distribution: Benign={test_class_dist[0]}, Attack={test_class_dist[1]}")

# === Step 8: Feature Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_temp)
X_test_scaled = scaler.transform(X_test_temp)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_temp.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_temp.values, dtype=torch.float32)

batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

logger.log("")
logger.log("=== Step 8: Feature Scaling and Data Preparation ===")
logger.log(f"Features scaled using StandardScaler")
logger.log(f"Batch size: {batch_size}")

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

input_dim = X_train_scaled.shape[1]
model = DDosDetectionNet(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

logger.log("")
logger.log("=== Step 9: Neural Network Architecture ===")
logger.log(f"Input dimension: {input_dim}")
logger.log(f"Architecture: {input_dim} -> 64 -> 32 -> 1")
logger.log(f"Activation: ReLU")
logger.log(f"Dropout: 0.3, 0.2")
logger.log(f"Output activation: Sigmoid")
logger.log(f"Loss function: Binary Cross Entropy")
logger.log(f"Optimizer: Adam (lr=0.001, weight_decay=1e-5)")

# === Step 10: Training Loop ===
logger.log("")
logger.log("=== Step 10: Training Neural Network ===")
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
        logger.log(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# Save training loss plot
plt.figure(figsize=(10, 6))
plt.plot(train_losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('plots/training_loss.png', dpi=300, bbox_inches='tight')
plt.close()

logger.log(f"Training completed. Loss plot saved to 'plots/training_loss.png'")
logger.log(f"Final training loss: {train_losses[-1]:.4f}")

# === Step 11: Cross-Validation ===
def pytorch_cv_score(X, y, n_splits=5):
    cv_scores = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_cv = X[train_idx]
        X_val_cv = X[val_idx]
        y_train_cv = y[train_idx]
        y_val_cv = y[val_idx]
        
        scaler_cv = StandardScaler()
        X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
        X_val_cv_scaled = scaler_cv.transform(X_val_cv)
        
        X_train_cv_tensor = torch.tensor(X_train_cv_scaled, dtype=torch.float32)
        y_train_cv_tensor = torch.tensor(y_train_cv, dtype=torch.float32)
        X_val_cv_tensor = torch.tensor(X_val_cv_scaled, dtype=torch.float32)
        
        model_cv = DDosDetectionNet(X_train_cv_scaled.shape[1])
        criterion_cv = nn.BCELoss()
        optimizer_cv = optim.Adam(model_cv.parameters(), lr=0.001, weight_decay=1e-5)
        
        model_cv.train()
        for epoch in range(20):
            optimizer_cv.zero_grad()
            outputs = model_cv(X_train_cv_tensor).squeeze()
            loss = criterion_cv(outputs, y_train_cv_tensor)
            loss.backward()
            optimizer_cv.step()
        
        model_cv.eval()
        with torch.no_grad():
            val_pred_probs = model_cv(X_val_cv_tensor).squeeze().numpy()
            cv_score = roc_auc_score(y_val_cv, val_pred_probs)
            cv_scores.append(cv_score)
    
    return np.array(cv_scores)

logger.log("")
logger.log("=== Step 11: Cross-Validation Results ===")
cv_scores = pytorch_cv_score(X_train_scaled, y_train_temp.values)
logger.log("5-Fold Cross-Validation ROC-AUC scores:")
for i, score in enumerate(cv_scores, 1):
    logger.log(f"  Fold {i}: {score:.4f}")
logger.log(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# === Step 12: Test Set Evaluation ===
logger.log("")
logger.log("=== Step 12: Test Set Results ===")
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_tensor).squeeze().numpy()
    y_preds = (y_pred_probs >= 0.5).astype(int)

cm = confusion_matrix(y_test_temp, y_preds)
cr = classification_report(y_test_temp, y_preds, output_dict=True)
roc_auc = roc_auc_score(y_test_temp, y_pred_probs)

logger.log("Confusion Matrix:")
logger.log(f"                Predicted")
logger.log(f"Actual    Benign  Attack")
logger.log(f"Benign    {cm[0][0]:6d}  {cm[0][1]:6d}")
logger.log(f"Attack    {cm[1][0]:6d}  {cm[1][1]:6d}")

logger.log("")
logger.log("Classification Report:")
logger.log(f"              precision    recall  f1-score   support")
logger.log(f"      Benign      {cr['0']['precision']:.2f}      {cr['0']['recall']:.2f}      {cr['0']['f1-score']:.2f}      {cr['0']['support']:.0f}")
logger.log(f"      Attack      {cr['1']['precision']:.2f}      {cr['1']['recall']:.2f}      {cr['1']['f1-score']:.2f}      {cr['1']['support']:.0f}")
logger.log(f"    accuracy                          {cr['accuracy']:.2f}      {len(y_test_temp)}")
logger.log(f"   macro avg      {cr['macro avg']['precision']:.2f}      {cr['macro avg']['recall']:.2f}      {cr['macro avg']['f1-score']:.2f}      {len(y_test_temp)}")
logger.log(f"weighted avg      {cr['weighted avg']['precision']:.2f}      {cr['weighted avg']['recall']:.2f}      {cr['weighted avg']['f1-score']:.2f}      {len(y_test_temp)}")

logger.log(f"\nROC-AUC Score: {roc_auc:.4f}")

# === Step 13: Random Split Comparison ===
logger.log("")
logger.log("=== Step 13: Random Split Comparison ===")
X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train_rand_scaled = scaler.fit_transform(X_train_rand)
X_test_rand_scaled = scaler.transform(X_test_rand)

X_train_rand_tensor = torch.tensor(X_train_rand_scaled, dtype=torch.float32)
y_train_rand_tensor = torch.tensor(y_train_rand.values, dtype=torch.float32)
X_test_rand_tensor = torch.tensor(X_test_rand_scaled, dtype=torch.float32)

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

model_rand.eval()
with torch.no_grad():
    y_pred_rand_probs = model_rand(X_test_rand_tensor).squeeze().numpy()
    y_pred_rand = (y_pred_rand_probs >= 0.5).astype(int)

cm_rand = confusion_matrix(y_test_rand, y_pred_rand)
roc_auc_rand = roc_auc_score(y_test_rand, y_pred_rand_probs)

logger.log("Random Split Confusion Matrix:")
logger.log(f"                Predicted")
logger.log(f"Actual    Benign  Attack")
logger.log(f"Benign    {cm_rand[0][0]:6d}  {cm_rand[0][1]:6d}")
logger.log(f"Attack    {cm_rand[1][0]:6d}  {cm_rand[1][1]:6d}")
logger.log(f"Random Split ROC-AUC: {roc_auc_rand:.4f}")

# === Step 14: Constant Attack Values Check ===
logger.log("")
logger.log("=== Step 14: Constant Attack Values Analysis ===")
constant_attack_features = []
for col in X.columns:
    attack_vals = X_temp[y_temp == 1][col].unique()
    if len(attack_vals) == 1:
        constant_attack_features.append(col)
        logger.log(f"Constant attack values: {col} = {attack_vals[0]}")

if not constant_attack_features:
    logger.log("No constant attack value features found.")

# === Step 15: Final Reality Check ===
tn, fp, fn, tp = cm.ravel()
logger.log("")
logger.log("=== Step 15: Final Reality Check ===")
logger.log("Real-world deployment implications:")
logger.log(f"- False Positive Rate: {fp/(fp+tn):.4f} ({fp} benign flows flagged as attacks)")
logger.log(f"- False Negative Rate: {fn/(fn+tp):.4f} ({fn} attacks missed)")
if fn > 0:
    logger.log(f"- This means missing 1 in {(tp+fn)//fn if fn > 0 else 'N/A'} DDoS attacks")
if fp > 0:
    logger.log(f"- And falsely alerting on 1 in {(tn+fp)//fp if fp > 0 else 'N/A'} benign flows")

# === Step 16: Save Models ===
logger.log("")
logger.log("=== Step 16: Model Saving ===")
joblib.dump(scaler, "models/ddos_scaler.pkl")
torch.save(model.state_dict(), "models/ddos_torch_model.pt")

model_info = {
    'input_dim': input_dim,
    'features': list(X.columns),
    'timestamp': timestamp,
    'performance': {
        'temporal_split_roc_auc': roc_auc,
        'random_split_roc_auc': roc_auc_rand,
        'cv_mean_roc_auc': cv_scores.mean(),
        'cv_std_roc_auc': cv_scores.std()
    }
}
joblib.dump(model_info, "models/ddos_model_info.pkl")

logger.log("Saved files:")
logger.log("- models/ddos_scaler.pkl")
logger.log("- models/ddos_torch_model.pt")
logger.log("- models/ddos_model_info.pkl")

# === Final Recommendations ===
logger.log("")
logger.log("=== Final Recommendations ===")
logger.log(" SUCCESS: PyTorch-based DDoS detection model completed!")
logger.log(" Neural network performance with meaningful trade-offs achieved")
logger.log("")
logger.log("🎯 Next steps for improvement:")
logger.log("1. Tune the decision threshold to balance precision/recall")
logger.log("2. Try different neural network architectures (deeper/wider networks)")
logger.log("3. Implement early stopping and learning rate scheduling")
logger.log("4. Add domain knowledge features (burst patterns, flow duration bins)")
logger.log("5. Test on different DDoS attack types")
logger.log("6. Consider ensemble methods combining multiple neural networks")
logger.log("7. Experiment with other PyTorch optimizers (SGD, RMSprop)")
logger.log("8. Add regularization techniques (batch normalization, different dropout rates)")

logger.log("")
logger.log(f"Analysis completed at: {datetime.now()}")

# === Generate PDF Report ===
def create_pdf_report():
    doc = SimpleDocTemplate(report_file, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        textColor=colors.darkblue
    )
    story.append(Paragraph("DDoS Detection Model Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    summary_data = [
        ['Metric', 'Value'],
        ['Dataset Shape (final)', f"{df.shape[0]} rows × {df.shape[1]} columns"],
        ['Features Used', str(len(X.columns))],
        ['Temporal Split ROC-AUC', f"{roc_auc:.4f}"],
        ['Cross-Validation ROC-AUC', f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"],
        ['False Positive Rate', f"{fp/(fp+tn):.4f}"],
        ['False Negative Rate', f"{fn/(fn+tp):.4f}"]
    ]
    
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    doc.build(story)

try:
    create_pdf_report()
    logger.log(f"\n PDF report generated: {report_file}")
except Exception as e:
    logger.log(f"\n PDF generation failed: {str(e)}")
    logger.log(" Complete analysis available in text log file")

logger.log(f"\n All outputs saved to:")
logger.log(f"  - Log file: {log_file}")
logger.log(f"  - Models: models/ directory")
logger.log(f"  - Plots: plots/ directory")
if os.path.exists(report_file):
    logger.log(f"  - PDF report: {report_file}")

print(f"Analysis complete! Check {log_file} for detailed results.")