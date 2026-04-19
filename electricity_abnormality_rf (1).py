"""
Electricity Abnormality Prediction — Random Forest Classifier
=============================================================
Graphs produced:
  1. Expected vs Actual Consumption
  2. Detection Status
  3. Energy Loss Trend
  4. Cluster-Based Analysis
  5. Feature Correlation Heatmap
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.facecolor']   = '#ffffff'
plt.rcParams['text.color']       = '#333333'
plt.rcParams['axes.labelcolor']  = '#333333'
plt.rcParams['xtick.color']      = '#555555'
plt.rcParams['ytick.color']      = '#555555'
plt.rcParams['axes.edgecolor']   = '#dddddd'
plt.rcParams['grid.color']       = '#eeeeee'
plt.rcParams['grid.linestyle']   = '--'
plt.rcParams['grid.alpha']       = 0.5
plt.rcParams['font.family']      = 'DejaVu Sans'

ACCENT  = '#00d4aa'   # teal    — normal class
ALERT   = '#ff6b6b'   # coral   — abnormal class
NEUTRAL = '#7c83fd'   # purple  — secondary
WARN    = '#ffd166'   # amber   — reference lines


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
import os
import joblib

print("Loading dataset...")
data_path = "data/electricity_data.csv"
if not os.path.exists(data_path):
    data_path = "Intelligent_abnormal_electricity_usage_dataset_REALWORLD.csv"

if not os.path.exists(data_path):
    print(f"Error: Dataset {data_path} not found.")
    exit(1)

df = pd.read_csv(data_path)
print(f"  Shape: {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLEAN DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Cleaning data...")

for col in ['Expected_Energy(kwh)', 'Actual_Energy(kwh)']:
    if col in df.columns:
        # 1. Convert to string and strip non-numeric (except dot)
        df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        # 2. Convert 'empty' or 'null' results to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # 3. Handle zero/NaN and ensure float
        df.loc[df[col] == 0, col] = np.nan
        mean_val = df[col].mean() if not df[col].isna().all() else 0
        df[col] = df[col].fillna(mean_val).astype(float).round(2)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("Engineering features...")

region_encoder = LabelEncoder()
dwelling_encoder = LabelEncoder()

df['Region_Code']   = region_encoder.fit_transform(df['Region_Code'].astype(str))
df['Dwelling_Type'] = dwelling_encoder.fit_transform(df['Dwelling_Type'].astype(str))

df['Deviation_Abs']    = abs(df['Actual_Energy(kwh)'] - df['Expected_Energy(kwh)'])

# Fix: Usage Ratio and Load Utilization with zero-checks
df['Usage_Ratio']      = np.where(df['Expected_Energy(kwh)'] != 0, 
                                 df['Actual_Energy(kwh)'] / df['Expected_Energy(kwh)'], 0)
df['Load_Utilization'] = np.where(df['Connected_Load(kw)'] != 0, 
                                 df['Actual_Energy(kwh)'] / df['Connected_Load(kw)'], 0)

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

df_graph = df.copy()

# Fix: dropped column handling
drop_cols = ['Meter_Id', 'Date', 'Expected_Energy(kwh)', 'Actual_Energy(kwh)',
             'Usage_Deviation(%)', 'Cluster_Avg_Energy(kwh)']
df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT & SCALING
# ══════════════════════════════════════════════════════════════════════════════
print("Splitting and scaling...")

if 'Abnormal_Usage' not in df.columns:
    print("Error: Abnormal_Usage column missing.")
    exit(1)

X = df.drop(['Abnormal_Usage'], axis=1)
y = df['Abnormal_Usage']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Save artifacts
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(region_encoder, "model/region_encoder.pkl")
joblib.dump(dwelling_encoder, "model/dwelling_encoder.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
print("Training Random Forest (100 trees)...")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf, "model/rf_model.pkl")

y_pred   = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"\n  Accuracy: {accuracy:.2f}%")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred))


# ══════════════════════════════════════════════════════════════════════════════
# 6. FLAG & HIGHLIGHT ABNORMAL METERS
# ══════════════════════════════════════════════════════════════════════════════

def flag_abnormal_meters(df_source, model, scaler, feature_cols):
    """
    Scans every row in the dataset, flags meters predicted as abnormal,
    assigns a Risk Level, and saves two outputs:
      - flagged_abnormal_meters.csv  : full flagged report sorted by severity
      - graph_flagged_highlight.png  : visual highlighting of abnormal meters

    Risk levels (based on Usage Ratio = Actual / Expected):
      CRITICAL  — ratio >= 2.0   (using 2x or more than expected)
      HIGH      — ratio >= 1.5
      MEDIUM    — ratio >= 1.2
      LOW       — flagged by model but ratio < 1.2 (subtle pattern)
    """

    print("Flagging abnormal meters...")

    df_flag = df_source.copy()

    # Predict on every row using the trained model
    X_all        = df_flag[feature_cols].values
    X_all_scaled = scaler.transform(X_all)

    df_flag['RF_Prediction']  = model.predict(X_all_scaled)
    df_flag['RF_Probability'] = model.predict_proba(X_all_scaled)[:, 1].round(4)
    df_flag['Overuse_kWh']    = (df_flag['Actual_Energy(kwh)'] - df_flag['Expected_Energy(kwh)']).round(2)

    # Assign risk level
    def assign_risk(row):
        if row['RF_Prediction'] != 1:
            return 'NORMAL'
        ratio = row.get('Usage_Ratio', 0)
        if ratio >= 2.0:   return 'CRITICAL'
        elif ratio >= 1.5: return 'HIGH'
        elif ratio >= 1.2: return 'MEDIUM'
        else:              return 'LOW'

    df_flag['Risk_Level'] = df_flag.apply(assign_risk, axis=1)

    # Pull out flagged rows only, sorted by severity then probability
    flagged = df_flag[df_flag['RF_Prediction'] == 1].copy()
    risk_order = pd.CategoricalDtype(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'], ordered=True)
    flagged['Risk_Level'] = flagged['Risk_Level'].astype(risk_order)
    flagged = flagged.sort_values(['Risk_Level', 'RF_Probability'], ascending=[True, False])

    # Save CSV report
    out_cols = [c for c in ['Meter_Id', 'Date', 'Region_Code', 'Dwelling_Type',
                             'Num_Occupants', 'Appliance_Score', 'Expected_Energy(kwh)',
                             'Actual_Energy(kwh)', 'Overuse_kWh', 'Usage_Ratio',
                             'RF_Probability', 'Risk_Level'] if c in flagged.columns]
    flagged[out_cols].to_csv('flagged_abnormal_meters.csv', index=False)

    # ── Print console report ──────────────────────────────────────────────────
    total         = len(df_flag)
    total_flagged = len(flagged)
    risk_counts   = flagged['Risk_Level'].value_counts()

    print(f"\n  {'-' * 48}")
    print(f"  ABNORMALITY FLAG REPORT")
    print(f"  {'-' * 48}")
    print(f"  Total readings scanned : {total:,}")
    print(f"  Flagged as ABNORMAL    : {total_flagged:,}  ({total_flagged / total * 100:.1f}%)")
    print(f"  {'-' * 48}")
    level_colors_console = {'CRITICAL': '[CRIT]', 'HIGH': '[HIGH]', 'MEDIUM': '[WARN]', 'LOW': '[INFO]'}
    for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = risk_counts.get(level, 0)
        print(f"  {level_colors_console[level]:<6} {level:<10} {count:>6,}")
    print(f"  {'-' * 48}")
    print(f"  Saved -> flagged_abnormal_meters.csv\n")

    print("  Top 10 highest-risk meters:")
    top_cols = [c for c in ['Meter_Id', 'Date', 'Expected_Energy(kwh)',
                             'Actual_Energy(kwh)', 'Overuse_kWh',
                             'RF_Probability', 'Risk_Level'] if c in flagged.columns]
    print(flagged[top_cols].head(10).to_string(index=False))
    print()

    # ── Highlight graph ───────────────────────────────────────────────────────
    risk_palette = {'CRITICAL': '#e63946', 'HIGH': '#ff6b6b',
                    'MEDIUM':   '#ffd166', 'LOW':  '#7c83fd'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Flagged Abnormal Meters — Highlight View',
                 fontsize=15, color='#333333', y=1.01)

    # Panel 1 — Scatter: Expected vs Actual, abnormals highlighted by risk level
    ax1 = axes[0]
    normal_pts = df_flag[df_flag['RF_Prediction'] == 0]
    ax1.scatter(normal_pts['Expected_Energy(kwh)'], normal_pts['Actual_Energy(kwh)'],
                color=ACCENT, alpha=0.25, s=12, edgecolors='none', label='Normal', zorder=1)
    for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
        pts = flagged[flagged['Risk_Level'] == level]
        if len(pts):
            ax1.scatter(pts['Expected_Energy(kwh)'], pts['Actual_Energy(kwh)'],
                        color=risk_palette[level], alpha=0.75, s=25,
                        edgecolors='none', label=level, zorder=2)
    lim = [df_flag['Expected_Energy(kwh)'].min(), df_flag['Expected_Energy(kwh)'].max()]
    ax1.plot(lim, lim, '--', color=WARN, linewidth=1, alpha=0.7)
    ax1.set_xlabel('Expected Energy (kWh)')
    ax1.set_ylabel('Actual Energy (kWh)')
    ax1.set_title('Flagged Meters — Consumption', color='#333333')
    ax1.legend(facecolor='#ffffff', edgecolor='#dddddd', labelcolor='#333333',
               fontsize=8, markerscale=1.5)
    ax1.grid(True)

    # Panel 2 — Risk level bar chart with counts
    ax2 = axes[1]
    levels       = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    counts_vals  = [risk_counts.get(l, 0) for l in levels]
    bar_colors   = [risk_palette[l] for l in levels]
    bars         = ax2.bar(levels, counts_vals, color=bar_colors, edgecolor='none', width=0.55)
    for bar, val in zip(bars, counts_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:,}', ha='center', va='bottom', fontsize=10, color='#333333')
    ax2.set_title('Flagged Count by Risk Level', color='#333333')
    ax2.set_ylabel('Number of Meter-Readings')
    ax2.grid(axis='y')

    # Panel 3 — Top 20 overusing meters, coloured by risk level
    ax3 = axes[2]
    top20      = flagged.nlargest(20, 'Overuse_kWh')
    top_colors = [risk_palette[str(r)] for r in top20['Risk_Level']]
    meter_labels = (top20['Meter_Id'].str[-8:]
                    if 'Meter_Id' in top20.columns
                    else top20.index.astype(str))
    ax3.barh(range(len(top20)), top20['Overuse_kWh'],
             color=top_colors, edgecolor='none')
    ax3.set_yticks(range(len(top20)))
    ax3.set_yticklabels(meter_labels, fontsize=7)
    ax3.invert_yaxis()
    ax3.set_xlabel('Overuse (kWh)')
    ax3.set_title('Top 20 Highest Overuse Meters', color='#333333')
    ax3.grid(axis='x')
    for i, val in enumerate(top20['Overuse_kWh']):
        ax3.text(val + 0.05, i, f'{val:.1f}', va='center', fontsize=7, color='#555555')

    # Shared legend for risk colours
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=risk_palette[l], label=l) for l in levels]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               facecolor='#ffffff', edgecolor='#dddddd', labelcolor='#333333',
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig('graph_flagged_highlight.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved -> graph_flagged_highlight.png\n")

    return flagged


# ── Gather the feature columns the model was trained on ──────────────────────
feature_cols = list(X.columns)   # X was defined in section 4

# ── Run flagging on the full dataset (df_graph still has all original columns)
flagged_df = flag_abnormal_meters(df_graph, rf, scaler, feature_cols)


# ══════════════════════════════════════════════════════════════════════════════
# 7. GRAPH 1 — Expected vs Actual Consumption
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting: Expected vs Actual Consumption...")

# Fix: check if df length is >= 300
sample_size = min(300, len(df_graph))
sample        = df_graph.sample(sample_size, random_state=42).reset_index(drop=True)
colors_scatter = [ALERT if v == 1 else ACCENT for v in sample['Abnormal_Usage']]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Expected vs Actual Energy Consumption', fontsize=15, color='#333333', y=1.01)

# Scatter
ax = axes[0]
ax.scatter(
    sample['Expected_Energy(kwh)'], sample['Actual_Energy(kwh)'],
    c=colors_scatter, alpha=0.7, s=30, edgecolors='none'
)
lim_min = min(sample['Expected_Energy(kwh)'].min(), sample['Actual_Energy(kwh)'].min())
lim_max = max(sample['Expected_Energy(kwh)'].max(), sample['Actual_Energy(kwh)'].max())
ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color=WARN, linewidth=1.2)
ax.set_xlabel('Expected Energy (kWh)')
ax.set_ylabel('Actual Energy (kWh)')
ax.set_title('Scatter: Expected vs Actual', color='#333333')
ax.grid(True)
ax.legend(
    handles=[
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor=ACCENT, markersize=8, label='Normal'),
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor=ALERT,  markersize=8, label='Abnormal'),
        mlines.Line2D([], [], color=WARN, linestyle='--', label='Perfect match'),
    ],
    facecolor='#ffffff', edgecolor='#dddddd', labelcolor='#333333'
)

# Box plot
ax2 = axes[1]
groups = [
    df_graph[df_graph['Abnormal_Usage'] == 0]['Expected_Energy(kwh)'],
    df_graph[df_graph['Abnormal_Usage'] == 1]['Expected_Energy(kwh)'],
    df_graph[df_graph['Abnormal_Usage'] == 0]['Actual_Energy(kwh)'],
    df_graph[df_graph['Abnormal_Usage'] == 1]['Actual_Energy(kwh)'],
]
bp = ax2.boxplot(
    groups,
    patch_artist=True,
    labels=['Expected\nNormal', 'Expected\nAbnormal', 'Actual\nNormal', 'Actual\nAbnormal'],
    medianprops=dict(color='black', linewidth=2)
)
for patch, color in zip(bp['boxes'], [ACCENT, ALERT, NEUTRAL, WARN]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_title('Distribution by Group', color='#333333')
ax2.set_ylabel('Energy (kWh)')
ax2.grid(axis='y')

plt.tight_layout()
plt.savefig('graph1_expected_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 8. GRAPH 2 — Detection Status
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting: Detection Status...")

pred_s = pd.Series(y_pred)
true_s = y_test.reset_index(drop=True)

tp = int(((pred_s == 1) & (true_s == 1)).sum())
tn = int(((pred_s == 0) & (true_s == 0)).sum())
fp = int(((pred_s == 1) & (true_s == 0)).sum())
fn = int(((pred_s == 0) & (true_s == 1)).sum())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Detection Status Analysis', fontsize=15, color='#333333', y=1.01)

# Pie — class distribution
ax = axes[0]
counts = df_graph['Abnormal_Usage'].value_counts()
wedges, texts, autotexts = ax.pie(
    counts.values,
    labels=['Normal', 'Abnormal'],
    autopct='%1.1f%%',
    colors=[ACCENT, ALERT],
    explode=[0, 0.07],
    startangle=140,
    wedgeprops=dict(edgecolor='#ffffff', linewidth=2)
)
for t in texts + autotexts:
    t.set_color('#333333')
ax.set_title('Class Distribution', color='#333333')

# Bar — prediction outcomes
ax2 = axes[1]
categories = ['True Positive\n(Correctly Abnormal)', 'True Negative\n(Correctly Normal)',
              'False Positive\n(Wrong Abnormal)',    'False Negative\n(Missed Abnormal)']
values     = [tp, tn, fp, fn]
bars       = ax2.bar(categories, values, color=[ALERT, ACCENT, WARN, NEUTRAL],
                     edgecolor='none', width=0.6)
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             str(val), ha='center', va='bottom', fontsize=11, color='#333333')
ax2.set_title('Prediction Outcomes', color='#333333')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', labelsize=8)
ax2.grid(axis='y')

# Bar — abnormality rate by dwelling type
ax3 = axes[2]
dt_grp = df_graph.groupby('Dwelling_Type')['Abnormal_Usage'].mean() * 100
dt_grp.plot(kind='bar', ax=ax3, color=ALERT, edgecolor='none', width=0.5)
ax3.set_title('Abnormality Rate by Dwelling Type', color='#333333')
ax3.set_ylabel('Abnormal %')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
ax3.grid(axis='y')

plt.tight_layout()
plt.savefig('graph2_detection_status.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 9. GRAPH 3 — Energy Loss Trend
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting: Energy Loss Trend...")

df_graph['Energy_Loss'] = df_graph['Actual_Energy(kwh)'] - df_graph['Expected_Energy(kwh)']

normal_loss   = df_graph[df_graph['Abnormal_Usage'] == 0]['Energy_Loss'].reset_index(drop=True)
abnormal_loss = df_graph[df_graph['Abnormal_Usage'] == 1]['Energy_Loss'].reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Energy Loss Trend', fontsize=15, color='#333333', y=1.01)

# Rolling mean
ax = axes[0]
window = 50
n_roll = normal_loss.rolling(window).mean()
a_roll = abnormal_loss.rolling(window).mean()
ax.plot(n_roll, color=ACCENT, linewidth=1.5, label='Normal (rolling avg)')
ax.plot(a_roll, color=ALERT,  linewidth=1.5, label='Abnormal (rolling avg)')
ax.axhline(0, color=WARN, linestyle='--', linewidth=1, alpha=0.7, label='Zero loss')
ax.fill_between(range(len(n_roll)), n_roll, 0, alpha=0.1, color=ACCENT)
ax.set_title(f'Rolling Mean Energy Loss (window={window})', color='#333333')
ax.set_xlabel('Record Index')
ax.set_ylabel('Energy Loss (kWh)')
ax.legend(facecolor='#ffffff', edgecolor='#dddddd', labelcolor='#333333')
ax.grid(True)

# KDE distribution
ax2 = axes[1]
sns.kdeplot(normal_loss,   ax=ax2, color=ACCENT, fill=True, alpha=0.35, label='Normal',   linewidth=2)
sns.kdeplot(abnormal_loss, ax=ax2, color=ALERT,  fill=True, alpha=0.35, label='Abnormal', linewidth=2)
ax2.axvline(0, color=WARN, linestyle='--', linewidth=1.2, alpha=0.8)
ax2.set_title('Energy Loss Distribution by Class', color='#333333')
ax2.set_xlabel('Energy Loss (kWh)')
ax2.set_ylabel('Density')
ax2.legend(facecolor='#ffffff', edgecolor='#dddddd', labelcolor='#333333')
ax2.grid(True)

plt.tight_layout()
plt.savefig('graph3_energy_loss_trend.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 10. GRAPH 4 — Cluster-Based Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting: Cluster-Based Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle('Cluster-Based Energy Analysis', fontsize=15, color='#333333', y=1.01)

# Cluster avg vs actual by region
ax = axes[0]
if 'Cluster_Avg_Energy(kwh)' in df_graph.columns:
    region_grp = df_graph.groupby('Region_Code').agg(
        cluster_avg=('Cluster_Avg_Energy(kwh)', 'mean'),
        actual_avg=('Actual_Energy(kwh)', 'mean')
    ).reset_index()
    x_pos = np.arange(len(region_grp))
    w     = 0.35
    ax.bar(x_pos - w/2, region_grp['cluster_avg'], width=w, color=NEUTRAL, alpha=0.8, label='Cluster Avg')
    ax.bar(x_pos + w/2, region_grp['actual_avg'],  width=w, color=ACCENT,  alpha=0.8, label='Actual Avg')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(region_grp['Region_Code'], rotation=45, ha='right', fontsize=8)
    ax.set_title('Cluster vs Actual Energy by Region', color='#333333')
    ax.set_ylabel('Energy (kWh)')
    ax.legend(facecolor='#ffffff', edgecolor='#dddddd', labelcolor='#333333')
else:
    region_grp = df_graph.groupby('Region_Code')['Actual_Energy(kwh)'].mean().reset_index()
    ax.bar(region_grp['Region_Code'].astype(str), region_grp['Actual_Energy(kwh)'],
           color=ACCENT, alpha=0.8, edgecolor='none')
    ax.set_title('Avg Actual Energy by Region', color='#333333')
    ax.set_ylabel('Energy (kWh)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.grid(axis='y')

# Abnormality rate by occupant count
ax2 = axes[1]
occ_abn = df_graph.groupby('Num_Occupants')['Abnormal_Usage'].mean() * 100
occ_cnt = df_graph.groupby('Num_Occupants')['Abnormal_Usage'].count()
ax2.bar(occ_abn.index.astype(str), occ_abn.values,
        color=WARN, alpha=0.85, edgecolor='none', width=0.6)
ax2_r = ax2.twinx()
ax2_r.plot(occ_cnt.index.astype(str), occ_cnt.values,
           color=NEUTRAL, marker='o', linewidth=1.5, markersize=5)
ax2_r.set_ylabel('Record Count', color=NEUTRAL)
ax2_r.tick_params(axis='y', labelcolor=NEUTRAL)
ax2.set_title('Abnormality Rate by Occupant Count', color='#333333')
ax2.set_xlabel('Number of Occupants')
ax2.set_ylabel('Abnormal %')
ax2.grid(axis='y')

# Deviation vs Usage Ratio scatter
ax3 = axes[2]
samp_size = min(500, len(df_graph))
samp = df_graph.sample(samp_size, random_state=7)
colors_cls = [ALERT if v == 1 else ACCENT for v in samp['Abnormal_Usage']]
ax3.scatter(samp['Deviation_Abs'], samp['Usage_Ratio'],
            c=colors_cls, alpha=0.55, s=20, edgecolors='none')
ax3.axhline(1.0, color=WARN, linestyle='--', linewidth=1, alpha=0.7)
ax3.set_xlabel('Deviation (Abs kWh)')
ax3.set_ylabel('Usage Ratio (Actual / Expected)')
ax3.set_title('Deviation vs Usage Ratio', color='#333333')
ax3.legend(
    handles=[
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor=ACCENT, markersize=8, label='Normal'),
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor=ALERT,  markersize=8, label='Abnormal'),
    ],
    facecolor='#ffffff', edgecolor='#dddddd', labelcolor='#333333'
)
ax3.grid(True)

plt.tight_layout()
plt.savefig('graph4_cluster_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 11. GRAPH 5 — Feature Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting: Feature Correlation Heatmap...")

numeric_cols = [
    'Expected_Energy(kwh)', 'Actual_Energy(kwh)',
    'Temperature_C', 'Num_Occupants', 'Appliance_Score',
    'Region_Code', 'Dwelling_Type',
    'Deviation_Abs', 'Usage_Ratio', 'Abnormal_Usage'
]
numeric_cols = [c for c in numeric_cols if c in df_graph.columns]
corr_matrix  = df_graph[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    ax=ax,
    mask=mask,
    cmap=sns.diverging_palette(240, 10, as_cmap=True),
    vmin=-1, vmax=1, center=0,
    annot=True, fmt='.2f',
    annot_kws={'size': 9, 'color': 'black'},
    linewidths=0.5,
    linecolor='#ffffff',
    square=True,
    cbar_kws={'shrink': 0.8, 'pad': 0.02}
)
ax.set_title('Feature Correlation Heatmap', fontsize=15, color='#333333', pad=16)
ax.tick_params(axis='x', rotation=45, labelsize=9)
ax.tick_params(axis='y', rotation=0,  labelsize=9)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(colors='#333333', labelsize=9)
cbar.outline.set_edgecolor('#dddddd')

plt.tight_layout()
plt.savefig('graph5_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 12. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 52)
print('  ELECTRICITY ABNORMALITY DETECTION - SUMMARY')
print('=' * 52)
print(f'  Model      : Random Forest Classifier')
print(f'  Trees      : 100  |  random_state=42')
print(f'  Train rows : {X_train.shape[0]}')
print(f'  Test rows  : {X_test.shape[0]}')
print(f'  Accuracy   : {accuracy:.2f}%')
print('=' * 52)
print()
print('  Outputs saved:')
print('    flagged_abnormal_meters.csv')
print('    graph_flagged_highlight.png')
print('    graph1_expected_vs_actual.png')
print('    graph2_detection_status.png')
print('    graph3_energy_loss_trend.png')
print('    graph4_cluster_analysis.png')
print('    graph5_heatmap.png')
print('=' * 52)
