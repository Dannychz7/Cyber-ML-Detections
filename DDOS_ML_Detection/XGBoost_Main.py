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
from scipy import stats
import logging
from datetime import datetime
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from config import DATAPATH_FRIDAY_WORK_HRS_AFTERNOON

# === Setup Directories and Logging ===
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"results/ddos_analysis_{timestamp}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Still show some output to console
    ]
)

logger = logging.getLogger(__name__)

class AnalysisResults:
    """Class to store analysis results for executive summary"""
    def __init__(self):
        self.dataset_shape = None
        self.features_removed = []
        self.final_feature_count = 0
        self.cv_auc_mean = 0
        self.cv_auc_std = 0
        self.temporal_auc = 0
        self.random_auc = 0
        self.false_positive_rate = 0
        self.false_negative_rate = 0
        self.top_features = []
        self.class_distribution = {}
        self.recommendations = []

results = AnalysisResults()

def log_section(title):
    """Helper function to log section headers"""
    logger.info("=" * 50)
    logger.info(f" {title}")
    logger.info("=" * 50)

def print_executive_summary():
    """Print executive summary to console"""
    print("\n" + "="*60)
    print("  DDOS DETECTION MODEL - EXECUTIVE SUMMARY")
    print("="*60)
    
    print(f"\n DATASET OVERVIEW")
    print(f"   • Original samples: {results.dataset_shape[0]:,}")
    print(f"   • Final features: {results.final_feature_count}")
    print(f"   • Features removed: {len(results.features_removed)}")
    
    print(f"\n MODEL PERFORMANCE")
    print(f"   • Cross-validation AUC: {results.cv_auc_mean:.3f} (±{results.cv_auc_std:.3f})")
    print(f"   • Temporal split AUC: {results.temporal_auc:.3f}")
    print(f"   • Random split AUC: {results.random_auc:.3f}")
    
    print(f"\n  ERROR RATES")
    print(f"   • False Positive Rate: {results.false_positive_rate:.4f}")
    print(f"   • False Negative Rate: {results.false_negative_rate:.4f}")
    
    print(f"\n TOP PREDICTIVE FEATURES")
    for i, (feature, importance) in enumerate(results.top_features[:5], 1):
        print(f"   {i}. {feature}: {importance:.3f}")
    
    print(f"\n RECOMMENDATIONS")
    for i, rec in enumerate(results.recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n Detailed logs saved to: {log_filename}")
    print("="*60)

def create_pdf_report():
    """Create a PDF report"""
    pdf_filename = f"results/ddos_analysis_report_{timestamp}.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.darkblue,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("DDoS Detection Model Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    
    summary_data = [
        ["Metric", "Value"],
        ["Dataset Size", f"{results.dataset_shape[0]:,} samples"],
        ["Final Feature Count", str(results.final_feature_count)],
        ["Features Removed", str(len(results.features_removed))],
        ["Cross-validation AUC", f"{results.cv_auc_mean:.3f} (±{results.cv_auc_std:.3f})"],
        ["Temporal Split AUC", f"{results.temporal_auc:.3f}"],
        ["Random Split AUC", f"{results.random_auc:.3f}"],
        ["False Positive Rate", f"{results.false_positive_rate:.4f}"],
        ["False Negative Rate", f"{results.false_negative_rate:.4f}"]
    ]
    
    table = Table(summary_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Top Features
    story.append(Paragraph("Top Predictive Features", styles['Heading2']))
    feature_data = [["Rank", "Feature", "Importance"]]
    for i, (feature, importance) in enumerate(results.top_features[:10], 1):
        feature_data.append([str(i), feature, f"{importance:.3f}"])
    
    feature_table = Table(feature_data)
    feature_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(feature_table)
    story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(Paragraph("Recommendations", styles['Heading2']))
    for i, rec in enumerate(results.recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
    
    doc.build(story)
    return pdf_filename

# === Main Analysis ===
log_section("STARTING DDOS DETECTION ANALYSIS")

# === Step 1: Load Data ===
logger.info("Loading dataset...")
csv_file = DATAPATH_FRIDAY_WORK_HRS_AFTERNOON
df = pd.read_csv(csv_file)
logger.info(f"Loaded dataset: {df.shape}")

# === Step 2: Clean Data ===
logger.info("Cleaning data...")
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
results.dataset_shape = df.shape
logger.info(f"Dataset shape after cleaning: {df.shape}")

# === Step 3: Binary Labeling ===
df['binary_label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
results.class_distribution = df['binary_label'].value_counts().to_dict()
logger.info(f"Class distribution: {results.class_distribution}")

# === Step 4: Remove Aggressive Leaky Features ===
log_section("FEATURE CLEANING")
df = df.drop(columns=['Label'])

leaky_features = [
    'Flow Bytes/s', 'Flow Packets/s', 'Init_Win_bytes_backward', 'Init_Win_bytes_forward',
    'Fwd Header Length', 'Bwd Header Length', 'Flow Duration', 'Total Fwd Packets', 
    'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packets/s', 'Bwd Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 
    'Flow IAT Min', 'Fwd IAT Total', 'Bwd IAT Total', 'Down/Up Ratio', 'Average Packet Size', 
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
]

removed_leaky = [col for col in leaky_features if col in df.columns]
df = df.drop(columns=removed_leaky)
results.features_removed.extend(removed_leaky)
logger.info(f"Removed {len(removed_leaky)} potentially leaky features")

# Drop constant columns
initial_cols = len(df.columns)
df = df.loc[:, df.nunique() > 1]
constant_cols_removed = initial_cols - len(df.columns)
logger.info(f"Dropped {constant_cols_removed} constant columns")

# === Step 5: Advanced Leakage Detection ===
log_section("ADVANCED LEAKAGE DETECTION")
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
        logger.info(f"Perfect separator found: {col}")
    elif len(overlap) > 0:
        total_unique = len(benign_vals | attack_vals)
        overlap_ratio = len(overlap) / total_unique
        if overlap_ratio < 0.05:
            near_perfect_separators.append((col, overlap_ratio))
            logger.info(f"Near-perfect separator: {col} (overlap: {overlap_ratio:.3f})")

# Remove problematic features
problematic_features = perfect_separators + [feat[0] for feat in near_perfect_separators]
if problematic_features:
    df = df.drop(columns=problematic_features)
    results.features_removed.extend(problematic_features)
    logger.info(f"Removed {len(problematic_features)} highly separating features")

# Check for constant attack values
constant_attack_features = []
for col in X_temp.columns:
    if col in df.columns:  # Make sure column still exists
        attack_vals = X_temp[y_temp == 1][col].unique()
        if len(attack_vals) == 1:
            constant_attack_features.append(col)
            logger.info(f"Constant attack values: {col} = {attack_vals[0]}")

if constant_attack_features:
    df = df.drop(columns=constant_attack_features)
    results.features_removed.extend(constant_attack_features)
    logger.info(f"Removed {len(constant_attack_features)} constant-attack-value features")

# === Step 6: Final Feature Setup ===
X = df.drop(columns=['binary_label'])
y = df['binary_label']
X.columns = [str(col) for col in X.columns]
results.final_feature_count = len(X.columns)

logger.info(f"Final feature count: {len(X.columns)}")
logger.info(f"Final features: {list(X.columns)}")

# Save final features list
with open("results/final_features.txt", "w") as f:
    for feature in X.columns:
        f.write(feature + "\n")

# === Step 7: Model Training and Evaluation ===
log_section("MODEL TRAINING AND EVALUATION")

# Temporal split
split_idx = int(len(df) * 0.7)
X_train_temp = X.iloc[:split_idx]
X_test_temp = X.iloc[split_idx:]
y_train_temp = y.iloc[:split_idx]
y_test_temp = y.iloc[split_idx:]

logger.info(f"Temporal split - Train: {len(X_train_temp)}, Test: {len(X_test_temp)}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_temp)
X_test_scaled = scaler.transform(X_test_temp)

# Train model
model = xgb.XGBClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.05,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, objective='binary:logistic',
    eval_metric='logloss', random_state=42
)

model.fit(X_train_scaled, y_train_temp)

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train_temp, 
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                           scoring='roc_auc')
results.cv_auc_mean = cv_scores.mean()
results.cv_auc_std = cv_scores.std()
logger.info(f"CV ROC-AUC: {results.cv_auc_mean:.4f} (+/- {results.cv_auc_std * 2:.4f})")

# Test set evaluation
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
results.temporal_auc = roc_auc_score(y_test_temp, y_pred_proba)

cm = confusion_matrix(y_test_temp, y_pred)
tn, fp, fn, tp = cm.ravel()
results.false_positive_rate = fp / (fp + tn)
results.false_negative_rate = fn / (fn + tp)

logger.info(f"Temporal split AUC: {results.temporal_auc:.4f}")
logger.info(f"False Positive Rate: {results.false_positive_rate:.4f}")
logger.info(f"False Negative Rate: {results.false_negative_rate:.4f}")

# Random split comparison
X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train_rand_scaled = scaler.fit_transform(X_train_rand)
X_test_rand_scaled = scaler.transform(X_test_rand)

model_rand = xgb.XGBClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.05,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, objective='binary:logistic',
    eval_metric='logloss', random_state=42
)

model_rand.fit(X_train_rand_scaled, y_train_rand)
y_pred_rand_proba = model_rand.predict_proba(X_test_rand_scaled)[:, 1]
results.random_auc = roc_auc_score(y_test_rand, y_pred_rand_proba)

logger.info(f"Random split AUC: {results.random_auc:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

results.top_features = list(zip(feature_importance['feature'].head(10), 
                               feature_importance['importance'].head(10)))

logger.info("Top 10 feature importances:")
for feature, importance in results.top_features:
    logger.info(f"  {feature}: {importance:.4f}")

# Create and save plots
plt.figure(figsize=(15, 10))

# Plot 1: Feature importance
plt.subplot(2, 2, 1)
top_features_df = feature_importance.head(10)
plt.barh(range(len(top_features_df)), top_features_df['importance'])
plt.yticks(range(len(top_features_df)), top_features_df['feature'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()

# Plot 2: Confusion Matrix
plt.subplot(2, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
plt.title('Confusion Matrix (Temporal Split)')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Plot 3: Class distribution
plt.subplot(2, 2, 3)
class_counts = list(results.class_distribution.values())
class_labels = ['Benign', 'Attack']
plt.pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=90)
plt.title('Class Distribution')

# Plot 4: AUC comparison
plt.subplot(2, 2, 4)
splits = ['Temporal', 'Random']
aucs = [results.temporal_auc, results.random_auc]
plt.bar(splits, aucs, color=['skyblue', 'lightcoral'])
plt.ylabel('AUC Score')
plt.title('AUC Comparison: Temporal vs Random Split')
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(f'plots/ddos_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()

# Save models
joblib.dump(scaler, "models/ddos_scaler_xgb.pkl")
joblib.dump(model_rand, "models/ddos_xgb_model.pkl")

# Generate recommendations
results.recommendations = [
    "Model shows realistic performance with meaningful trade-offs",
    "Consider ensemble methods (Random Forest + XGBoost) for improvement",
    "Tune decision threshold to optimize precision/recall balance",
    "Test on different DDoS attack types for robustness",
    "Consider anomaly detection for unknown attack variants",
    "Add domain knowledge features (burst patterns, flow duration bins)",
    "Monitor model performance in production environment"
]

# Create PDF report
pdf_file = create_pdf_report()
logger.info(f"PDF report saved to: {pdf_file}")

log_section("ANALYSIS COMPLETE")
logger.info("All outputs saved:")
logger.info(f"- Detailed log: {log_filename}")
logger.info(f"- PDF report: {pdf_file}")
logger.info(f"- Feature list: results/final_features.txt")
logger.info(f"- Visualization: plots/ddos_analysis_{timestamp}.png")
logger.info("- Models: models/ddos_scaler.pkl, models/ddos_xgb_model.pkl")

# Print executive summary to console
print_executive_summary()