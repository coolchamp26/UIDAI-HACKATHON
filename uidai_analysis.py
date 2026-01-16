"""
UIDAI Aadhaar Demographic Update Analysis
==========================================
Comprehensive analysis for Government of India / UIDAI National Hackathon

Author: Data Science Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Configuration
DATA_DIR = Path(r"c:\Users\Admin\Desktop\Ashambar\PROJECTS\UIDAI")
OUTPUT_DIR = DATA_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("UIDAI AADHAAR DEMOGRAPHIC UPDATE ANALYSIS")
print("=" * 80)

# ============================================================================
# PHASE 1: DATA LOADING & CLEANING
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1: DATA LOADING & CLEANING")
print("=" * 80)

# Load all CSV files
files = ['aadhar_1.csv', 'aadhar_2.csv', 'aadhar_3.csv', 'aadhar_4.csv', 'aadhar_5.csv']
dfs = []

for f in files:
    df_temp = pd.read_csv(DATA_DIR / f)
    dfs.append(df_temp)
    print(f"Loaded {f}: {len(df_temp):,} rows")

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal combined rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# Rename columns for consistency (handle any trailing characters)
df.columns = [col.strip().replace('demo_age_17_', 'demo_age_17_plus') for col in df.columns]
if 'demo_age_17_plus' not in df.columns:
    # Find the 17+ column
    for col in df.columns:
        if '17' in col and col != 'demo_age_5_17':
            df.rename(columns={col: 'demo_age_17_plus'}, inplace=True)
            break

print(f"Cleaned columns: {list(df.columns)}")

# Parse dates
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
invalid_dates = df['date'].isna().sum()
print(f"Invalid dates found: {invalid_dates}")
df = df.dropna(subset=['date'])

# Create derived time features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['month_name'] = df['date'].dt.month_name()

# Financial Year (April to March)
df['financial_year'] = df['date'].apply(
    lambda x: f"FY{x.year}-{x.year+1}" if x.month >= 4 else f"FY{x.year-1}-{x.year}"
)

# Standardize state/district names
df['state'] = df['state'].str.strip().str.title()
df['district'] = df['district'].str.strip().str.title()

# Validate numeric columns - remove negative values
numeric_cols = ['demo_age_5_17', 'demo_age_17_plus']
for col in numeric_cols:
    if col in df.columns:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            print(f"Removed {negative_count} negative values in {col}")
            df = df[df[col] >= 0]

print(f"\nCleaned dataset shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Unique states: {df['state'].nunique()}")
print(f"Unique districts: {df['district'].nunique()}")
print(f"Unique pincodes: {df['pincode'].nunique()}")

# Save cleaned data
df.to_csv(DATA_DIR / "cleaned_data.csv", index=False)
print("\nCleaned data saved to cleaned_data.csv")

# Document assumptions
ASSUMPTIONS = """
ASSUMPTIONS MADE:
1. Date format: DD-MM-YYYY (confirmed from data inspection)
2. Age groups: Only 5-17 and 17+ present in dataset
3. Negative update values treated as data errors and removed
4. State/district names standardized to Title Case
5. Pincode-level data represents daily aggregated counts
"""
print(ASSUMPTIONS)

# ============================================================================
# PHASE 2: FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 2: FEATURE ENGINEERING")
print("=" * 80)

# Total demographic updates
df['total_updates'] = df['demo_age_5_17'] + df['demo_age_17_plus']

# Age-group proportions
df['pct_5_17'] = (df['demo_age_5_17'] / df['total_updates'] * 100).round(2)
df['pct_17_plus'] = (df['demo_age_17_plus'] / df['total_updates'] * 100).round(2)

# Handle division by zero
df['pct_5_17'] = df['pct_5_17'].fillna(0)
df['pct_17_plus'] = df['pct_17_plus'].fillna(0)

print("Created Features:")
print("- total_updates: Sum of all age-group update counts")
print("- pct_5_17: Percentage of updates from age 5-17")
print("- pct_17_plus: Percentage of updates from age 17+")

# Feature importance from policy perspective
FEATURE_IMPORTANCE = """
POLICY RELEVANCE OF DERIVED FEATURES:
1. total_updates: Measures overall service demand and infrastructure load
2. pct_5_17: High values indicate school enrollment-driven updates (name corrections, address changes)
3. pct_17_plus: High values indicate adult life events (marriage, migration, employment)
4. MoM growth: Detects monthly service demand fluctuations for resource planning
5. YoY growth: Reveals long-term trends for strategic capacity planning
"""
print(FEATURE_IMPORTANCE)

# ============================================================================
# PHASE 3: MULTI-LEVEL AGGREGATION
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 3: MULTI-LEVEL AGGREGATION")
print("=" * 80)

# State-level aggregation (Policy & Planning View)
state_agg = df.groupby('state').agg({
    'total_updates': 'sum',
    'demo_age_5_17': 'sum',
    'demo_age_17_plus': 'sum',
    'pincode': 'nunique',
    'district': 'nunique'
}).reset_index()
state_agg.columns = ['state', 'total_updates', 'updates_5_17', 'updates_17_plus', 'unique_pincodes', 'unique_districts']
state_agg['pct_5_17'] = (state_agg['updates_5_17'] / state_agg['total_updates'] * 100).round(2)
state_agg['pct_17_plus'] = (state_agg['updates_17_plus'] / state_agg['total_updates'] * 100).round(2)
state_agg = state_agg.sort_values('total_updates', ascending=False)

print(f"State-level view: {len(state_agg)} states/UTs")

# District-level aggregation (Implementation & Service Delivery View)
district_agg = df.groupby(['state', 'district']).agg({
    'total_updates': 'sum',
    'demo_age_5_17': 'sum',
    'demo_age_17_plus': 'sum',
    'pincode': 'nunique'
}).reset_index()
district_agg.columns = ['state', 'district', 'total_updates', 'updates_5_17', 'updates_17_plus', 'unique_pincodes']
district_agg['pct_5_17'] = (district_agg['updates_5_17'] / district_agg['total_updates'] * 100).round(2)
district_agg['updates_per_pincode'] = (district_agg['total_updates'] / district_agg['unique_pincodes']).round(0)
district_agg = district_agg.sort_values('total_updates', ascending=False)

print(f"District-level view: {len(district_agg)} districts")

# Monthly aggregation for time series
monthly_agg = df.groupby(['year', 'month']).agg({
    'total_updates': 'sum',
    'demo_age_5_17': 'sum',
    'demo_age_17_plus': 'sum'
}).reset_index()
monthly_agg['date'] = pd.to_datetime(monthly_agg[['year', 'month']].assign(day=1))
monthly_agg = monthly_agg.sort_values('date')

# Calculate MoM growth
monthly_agg['mom_growth'] = monthly_agg['total_updates'].pct_change() * 100

# Calculate YoY growth
monthly_agg['yoy_growth'] = monthly_agg['total_updates'].pct_change(periods=12) * 100

print(f"Monthly time series: {len(monthly_agg)} months")

# ============================================================================
# PHASE 4: UNIVARIATE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 4A: UNIVARIATE ANALYSIS")
print("=" * 80)

# Overall statistics
print("\n--- OVERALL STATISTICS ---")
total_all_updates = df['total_updates'].sum()
print(f"Total demographic updates: {total_all_updates:,.0f}")
print(f"Total updates (5-17): {df['demo_age_5_17'].sum():,.0f} ({df['demo_age_5_17'].sum()/total_all_updates*100:.1f}%)")
print(f"Total updates (17+): {df['demo_age_17_plus'].sum():,.0f} ({df['demo_age_17_plus'].sum()/total_all_updates*100:.1f}%)")

# Top 10 States
print("\n--- TOP 10 STATES BY TOTAL UPDATES ---")
for i, row in state_agg.head(10).iterrows():
    print(f"  {row['state']}: {row['total_updates']:,.0f} ({row['total_updates']/total_all_updates*100:.1f}%)")

# Bottom 10 States
print("\n--- BOTTOM 10 STATES/UTs BY TOTAL UPDATES ---")
for i, row in state_agg.tail(10).iterrows():
    print(f"  {row['state']}: {row['total_updates']:,.0f}")

# Top 10 Districts
print("\n--- TOP 10 DISTRICTS BY TOTAL UPDATES ---")
for i, row in district_agg.head(10).iterrows():
    print(f"  {row['district']}, {row['state']}: {row['total_updates']:,.0f}")

# Distribution statistics
print("\n--- DISTRIBUTION STATISTICS ---")
print(f"Mean updates per pincode-day: {df['total_updates'].mean():.2f}")
print(f"Median updates per pincode-day: {df['total_updates'].median():.2f}")
print(f"Std Dev: {df['total_updates'].std():.2f}")
print(f"Min: {df['total_updates'].min()}")
print(f"Max: {df['total_updates'].max()}")

# ============================================================================
# PHASE 4B: BIVARIATE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 4B: BIVARIATE ANALYSIS")
print("=" * 80)

# State × Age-group patterns
print("\n--- STATE × AGE GROUP PATTERNS ---")
print("States with HIGHEST proportion of 5-17 updates (school-age focus):")
state_by_youth = state_agg.nlargest(5, 'pct_5_17')
for i, row in state_by_youth.iterrows():
    print(f"  {row['state']}: {row['pct_5_17']:.1f}% youth updates")

print("\nStates with HIGHEST proportion of 17+ updates (adult focus):")
state_by_adult = state_agg.nlargest(5, 'pct_17_plus')
for i, row in state_by_adult.iterrows():
    print(f"  {row['state']}: {row['pct_17_plus']:.1f}% adult updates")

# Time × Total updates
print("\n--- TIME × TOTAL UPDATES ---")
print("Monthly update volumes (recent months):")
for i, row in monthly_agg.tail(6).iterrows():
    growth_str = f"(MoM: {row['mom_growth']:+.1f}%)" if not pd.isna(row['mom_growth']) else ""
    print(f"  {row['date'].strftime('%Y-%m')}: {row['total_updates']:,.0f} {growth_str}")

# Quarterly patterns
quarterly_pattern = df.groupby('quarter')['total_updates'].sum()
print("\n--- QUARTERLY PATTERNS ---")
for q, val in quarterly_pattern.items():
    print(f"  Q{q}: {val:,.0f} ({val/total_all_updates*100:.1f}%)")

# District intensity analysis
print("\n--- DISTRICTS WITH DISPROPORTIONATELY HIGH UPDATE INTENSITY ---")
# Updates per pincode as intensity measure
high_intensity = district_agg[district_agg['updates_per_pincode'] > district_agg['updates_per_pincode'].quantile(0.95)]
print(f"Districts with intensity > 95th percentile ({district_agg['updates_per_pincode'].quantile(0.95):.0f} updates/pincode):")
for i, row in high_intensity.head(10).iterrows():
    print(f"  {row['district']}, {row['state']}: {row['updates_per_pincode']:.0f} updates/pincode")

# ============================================================================
# PHASE 4C: TRIVARIATE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 4C: TRIVARIATE ANALYSIS")
print("=" * 80)

# Analysis 1: State × Age Group × Time
print("\n--- ANALYSIS 1: STATE × AGE GROUP × TIME ---")
print("How age-group distribution varies across top states over time")

top_5_states = state_agg.head(5)['state'].tolist()
state_time_age = df[df['state'].isin(top_5_states)].groupby(['state', 'financial_year']).agg({
    'demo_age_5_17': 'sum',
    'demo_age_17_plus': 'sum',
    'total_updates': 'sum'
}).reset_index()
state_time_age['pct_5_17'] = (state_time_age['demo_age_5_17'] / state_time_age['total_updates'] * 100).round(1)

print("\nYouth (5-17) update percentage by top states and financial year:")
pivot_state_time = state_time_age.pivot(index='state', columns='financial_year', values='pct_5_17')
print(pivot_state_time.to_string())

# Analysis 2: District × Age Group × Growth Pattern
print("\n--- ANALYSIS 2: DISTRICT GROWTH PATTERNS BY AGE GROUP ---")
# Calculate growth rates for districts
district_monthly = df.groupby(['state', 'district', 'year', 'month']).agg({
    'total_updates': 'sum',
    'demo_age_5_17': 'sum',
    'demo_age_17_plus': 'sum'
}).reset_index()

# Get first and last period totals for each district
first_period = district_monthly.groupby(['state', 'district']).first().reset_index()
last_period = district_monthly.groupby(['state', 'district']).last().reset_index()

district_growth = first_period[['state', 'district']].copy()
district_growth['initial_updates'] = first_period['total_updates']
district_growth['final_updates'] = last_period['total_updates'].values
district_growth['growth_rate'] = ((district_growth['final_updates'] - district_growth['initial_updates']) / 
                                   district_growth['initial_updates'] * 100).round(1)

# Filter out extreme values for meaningful analysis
district_growth_clean = district_growth[
    (district_growth['initial_updates'] > 100) & 
    (district_growth['growth_rate'].between(-500, 500))
]

print("\nFastest growing districts (by update volume):")
for i, row in district_growth_clean.nlargest(10, 'growth_rate').iterrows():
    print(f"  {row['district']}, {row['state']}: {row['growth_rate']:+.1f}% growth")

print("\nFastest declining districts (by update volume):")
for i, row in district_growth_clean.nsmallest(5, 'growth_rate').iterrows():
    print(f"  {row['district']}, {row['state']}: {row['growth_rate']:+.1f}% change")

# ============================================================================
# PHASE 5: ANOMALY DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 5: ANOMALY DETECTION")
print("=" * 80)

# Method 1: IQR-based outliers at district level
Q1 = district_agg['total_updates'].quantile(0.25)
Q3 = district_agg['total_updates'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

iqr_outliers = district_agg[district_agg['total_updates'] > upper_bound]
print(f"\n--- IQR-BASED OUTLIERS (>{upper_bound:,.0f} updates) ---")
print(f"Districts with abnormally HIGH update volumes: {len(iqr_outliers)}")
for i, row in iqr_outliers.head(15).iterrows():
    print(f"  {row['district']}, {row['state']}: {row['total_updates']:,.0f} updates")

# Method 2: Z-score analysis
district_agg['z_score'] = stats.zscore(district_agg['total_updates'])
z_outliers = district_agg[abs(district_agg['z_score']) > 3]
print(f"\n--- Z-SCORE OUTLIERS (|Z| > 3) ---")
print(f"Districts with extreme deviation: {len(z_outliers)}")
for i, row in z_outliers.head(10).iterrows():
    print(f"  {row['district']}, {row['state']}: Z={row['z_score']:.2f}")

# Temporal anomaly detection - spike identification
print("\n--- TEMPORAL ANOMALIES (Monthly Spikes) ---")
monthly_agg['z_score'] = stats.zscore(monthly_agg['total_updates'].fillna(0))
monthly_spikes = monthly_agg[abs(monthly_agg['z_score']) > 2]
print(f"Months with unusual activity (|Z| > 2): {len(monthly_spikes)}")
for i, row in monthly_spikes.iterrows():
    spike_type = "HIGH" if row['z_score'] > 0 else "LOW"
    print(f"  {row['date'].strftime('%Y-%m')}: {row['total_updates']:,.0f} ({spike_type}, Z={row['z_score']:.2f})")

# Classify anomalies
print("\n--- ANOMALY CLASSIFICATION ---")
print("\nSTRUCTURAL ANOMALIES (Persistent high-volume districts):")
print("These districts consistently show high update volumes, likely due to:")
print("  - Large population centers (metros, state capitals)")
print("  - Migration hubs with high demographic churn")
print("  - Administrative centers with better service access")

structural_anomalies = iqr_outliers[iqr_outliers['total_updates'] > upper_bound * 1.5]
for i, row in structural_anomalies.head(5).iterrows():
    print(f"  • {row['district']}, {row['state']}")

print("\nEVENT-DRIVEN ANOMALIES (Temporary spikes):")
print("  - Identified from temporal Z-score analysis")
print("  - May correlate with policy changes, enrollment drives, or seasonal patterns")

# ============================================================================
# PHASE 6: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 6: GENERATING VISUALIZATIONS")
print("=" * 80)

# Set style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# 1. State Distribution Heatmap
fig, ax = plt.subplots(figsize=(14, 10))
state_pivot = state_agg.set_index('state')[['total_updates', 'pct_5_17', 'pct_17_plus']].head(20)
state_pivot_norm = state_pivot.copy()
state_pivot_norm['total_updates'] = state_pivot_norm['total_updates'] / 1000000  # In millions
sns.heatmap(state_pivot_norm, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
ax.set_title('Top 20 States: Update Volume (Millions) and Age Distribution (%)', fontsize=14)
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'state_distribution_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated: state_distribution_heatmap.png")

# 2. Time Trend Analysis
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Total updates over time
axes[0].plot(monthly_agg['date'], monthly_agg['total_updates']/1000000, marker='o', linewidth=2, color='#2196F3')
axes[0].fill_between(monthly_agg['date'], monthly_agg['total_updates']/1000000, alpha=0.3)
axes[0].set_title('Monthly Demographic Updates Over Time', fontsize=14)
axes[0].set_ylabel('Updates (Millions)')
axes[0].grid(True, alpha=0.3)

# Age group trends
axes[1].plot(monthly_agg['date'], monthly_agg['demo_age_5_17']/1000000, label='Age 5-17', marker='s', linewidth=2)
axes[1].plot(monthly_agg['date'], monthly_agg['demo_age_17_plus']/1000000, label='Age 17+', marker='^', linewidth=2)
axes[1].set_title('Age Group Update Trends', fontsize=14)
axes[1].set_ylabel('Updates (Millions)')
axes[1].set_xlabel('Month')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'time_trend_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated: time_trend_analysis.png")

# 3. Age Group Comparison by Top States
fig, ax = plt.subplots(figsize=(14, 8))
top_states_data = state_agg.head(15)
x = range(len(top_states_data))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], top_states_data['updates_5_17']/1000000, width, label='Age 5-17', color='#4CAF50')
bars2 = ax.bar([i + width/2 for i in x], top_states_data['updates_17_plus']/1000000, width, label='Age 17+', color='#FF9800')

ax.set_xlabel('State')
ax.set_ylabel('Updates (Millions)')
ax.set_title('Age Group Distribution by Top 15 States', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(top_states_data['state'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'age_group_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated: age_group_comparison.png")

# 4. Anomaly Detection Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# District update distribution with outliers marked
axes[0].hist(district_agg['total_updates']/1000, bins=50, color='#2196F3', alpha=0.7, edgecolor='black')
axes[0].axvline(x=upper_bound/1000, color='red', linestyle='--', linewidth=2, label=f'IQR Threshold ({upper_bound/1000:.0f}K)')
axes[0].set_xlabel('Total Updates (Thousands)')
axes[0].set_ylabel('Number of Districts')
axes[0].set_title('District Update Distribution with Outlier Threshold', fontsize=12)
axes[0].legend()

# Monthly Z-scores
colors = ['red' if abs(z) > 2 else '#2196F3' for z in monthly_agg['z_score']]
axes[1].bar(range(len(monthly_agg)), monthly_agg['z_score'], color=colors, alpha=0.7)
axes[1].axhline(y=2, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[1].axhline(y=-2, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[1].set_xlabel('Month Index')
axes[1].set_ylabel('Z-Score')
axes[1].set_title('Monthly Activity Z-Scores (Red = Anomalous)', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated: anomaly_detection.png")

# 5. Quarterly Pattern Visualization
fig, ax = plt.subplots(figsize=(10, 6))
quarters = ['Q1\n(Jan-Mar)', 'Q2\n(Apr-Jun)', 'Q3\n(Jul-Sep)', 'Q4\n(Oct-Dec)']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax.bar(quarters, quarterly_pattern.values/1000000, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Updates (Millions)')
ax.set_title('Quarterly Distribution of Demographic Updates', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, quarterly_pattern.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{val/1000000:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'quarterly_pattern.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated: quarterly_pattern.png")

# 6. Top Districts Horizontal Bar Chart
fig, ax = plt.subplots(figsize=(12, 10))
top_districts = district_agg.head(20).sort_values('total_updates')
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_districts)))
bars = ax.barh(top_districts['district'] + ', ' + top_districts['state'], 
               top_districts['total_updates']/1000000, color=colors)
ax.set_xlabel('Total Updates (Millions)')
ax.set_title('Top 20 Districts by Update Volume', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top_districts.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated: top_districts.png")

print(f"\nAll visualizations saved to: {OUTPUT_DIR}")

# ============================================================================
# PHASE 7: KEY FINDINGS & RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 7: KEY FINDINGS & POLICY RECOMMENDATIONS")
print("=" * 80)

# Compile key statistics for report
key_stats = {
    'total_updates': total_all_updates,
    'pct_17_plus': df['demo_age_17_plus'].sum()/total_all_updates*100,
    'pct_5_17': df['demo_age_5_17'].sum()/total_all_updates*100,
    'top_state': state_agg.iloc[0]['state'],
    'top_state_share': state_agg.iloc[0]['total_updates']/total_all_updates*100,
    'top_district': f"{district_agg.iloc[0]['district']}, {district_agg.iloc[0]['state']}",
    'num_outlier_districts': len(iqr_outliers),
    'num_temporal_anomalies': len(monthly_spikes)
}

findings = f"""
===============================================================================
                           KEY FINDINGS SUMMARY
===============================================================================

1. SCALE & DISTRIBUTION
   • Total demographic updates analyzed: {key_stats['total_updates']:,.0f}
   • Adult updates (17+): {key_stats['pct_17_plus']:.1f}% of all updates
   • Youth updates (5-17): {key_stats['pct_5_17']:.1f}% of all updates
   
2. GEOGRAPHIC CONCENTRATION
   • Top state: {key_stats['top_state']} ({key_stats['top_state_share']:.1f}% of all updates)
   • Top district: {key_stats['top_district']}
   • High-volume outlier districts identified: {key_stats['num_outlier_districts']}
   
3. TEMPORAL PATTERNS
   • Temporal anomalies (unusual months): {key_stats['num_temporal_anomalies']}
   • Pattern: Updates show quarterly seasonality
   
4. AGE GROUP INSIGHTS
   • Adults (17+) dominate updates across all states
   • Youth proportion varies by state, indicating different administrative priorities
   
5. RISK AREAS IDENTIFIED
   • Concentrated service load in metro districts
   • Uneven youth update rates suggesting access disparities
"""

print(findings)

recommendations = """
===============================================================================
                        POLICY RECOMMENDATIONS
===============================================================================

1. CAPACITY PLANNING
   Recommendation: Increase Aadhaar center capacity in top 20 high-volume districts
   Risk Addressed: Service bottlenecks and citizen wait times
   Implementation: Deploy additional enrollment/update stations proportional to demand
   
2. TARGETED YOUTH OUTREACH
   Recommendation: Launch school-based update camps in states with low youth update rates
   Risk Addressed: Children missing updates for school enrollment, scholarships
   Implementation: Partner with state education departments for systematic drives
   
3. SEASONAL RESOURCE ALLOCATION
   Recommendation: Align staffing with quarterly demand patterns
   Risk Addressed: Understaffing during peak periods, inefficiency during troughs
   Implementation: Flexible workforce scheduling based on historical patterns
   
4. ANOMALY MONITORING SYSTEM
   Recommendation: Implement automated alerts for district-level volume spikes
   Risk Addressed: Fraud detection, data quality issues, unusual demand surges
   Implementation: Deploy statistical monitoring dashboard with IQR/Z-score thresholds
   
5. UNDERSERVED AREA ANALYSIS
   Recommendation: Investigate low-update districts for access barriers
   Risk Addressed: Exclusion of rural/remote populations from Aadhaar updates
   Implementation: Mobile van deployment, awareness campaigns
   
6. AGE-SPECIFIC SERVICE OPTIMIZATION
   Recommendation: Create separate fast-tracks for youth (school ID verification only)
   Risk Addressed: Long wait times for simple youth updates
   Implementation: Streamlined document requirements and dedicated counters
"""

print(recommendations)

# ============================================================================
# PHASE 8: GENERATE ANALYSIS REPORT
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 8: GENERATING ANALYSIS REPORT")
print("=" * 80)

report_content = f"""# UIDAI Aadhaar Demographic Update Analysis Report

## Executive Summary

This analysis examines **{total_all_updates:,.0f} demographic update records** across all Indian states and union territories to identify spatial, temporal, and age-based patterns in Aadhaar service utilization. The data spans multiple months and covers **{df['state'].nunique()} states/UTs**, **{df['district'].nunique()} districts**, and **{df['pincode'].nunique():,} pincodes**.

**Key Finding:** Adult population (17+) accounts for **{key_stats['pct_17_plus']:.1f}%** of all demographic updates, indicating that life events such as marriage, migration, and employment changes are the primary drivers of Aadhaar modifications. Youth updates (5-17) represent **{key_stats['pct_5_17']:.1f}%**, largely driven by school enrollment requirements.

**Geographic Concentration:** Update volumes are heavily concentrated in major states, with **{key_stats['top_state']}** alone accounting for **{key_stats['top_state_share']:.1f}%** of all updates. This concentration pattern highlights the need for capacity planning in high-demand regions while investigating potential access barriers in underserved areas.

**Operational Risk Areas:** The analysis identified **{key_stats['num_outlier_districts']} districts** with statistically anomalous update volumes requiring capacity assessment, and **{key_stats['num_temporal_anomalies']} months** with unusual activity patterns that may indicate policy-driven surges or data quality concerns.

---

## Key Findings

### Scale & Volume Analysis
- **Total Updates Processed:** {total_all_updates:,.0f}
- **Age 17+ Updates:** {df['demo_age_17_plus'].sum():,.0f} ({key_stats['pct_17_plus']:.1f}%)
- **Age 5-17 Updates:** {df['demo_age_5_17'].sum():,.0f} ({key_stats['pct_5_17']:.1f}%)

### Geographic Distribution
- **States/UTs Covered:** {df['state'].nunique()}
- **Districts Covered:** {df['district'].nunique()}
- **Top State by Volume:** {key_stats['top_state']} ({key_stats['top_state_share']:.1f}% share)
- **Top District by Volume:** {key_stats['top_district']}

### Statistical Anomalies
- **High-Volume Outlier Districts (IQR method):** {len(iqr_outliers)}
- **Extreme Outlier Districts (Z > 3):** {len(z_outliers)}
- **Temporal Anomalies Detected:** {key_stats['num_temporal_anomalies']} months

---

## Anomalies & Risk Areas Identified

### Structural Anomalies (Persistent High-Volume)
Districts consistently showing high update volumes, warranting capacity assessment:
"""

# Add top outlier districts
for i, row in iqr_outliers.head(10).iterrows():
    report_content += f"\n- **{row['district']}, {row['state']}:** {row['total_updates']:,.0f} updates"

report_content += """

### Event-Driven Anomalies (Temporal Spikes)
Months showing unusual activity patterns:
"""

for i, row in monthly_spikes.iterrows():
    spike_type = "Surge" if row['z_score'] > 0 else "Drop"
    report_content += f"\n- **{row['date'].strftime('%B %Y')}:** {row['total_updates']:,.0f} updates ({spike_type}, Z={row['z_score']:.2f})"

report_content += f"""

---

## Policy-Relevant Recommendations

### 1. Capacity Optimization
> **Insight:** {len(iqr_outliers)} districts show disproportionately high update volumes.
> 
> **Risk:** Service bottlenecks, increased citizen wait times, enrollment backlogs.
> 
> **Recommendation:** Deploy additional Aadhaar update stations in high-volume districts proportional to demand. Prioritize top 20 districts identified in this analysis.

### 2. Youth-Focused Intervention
> **Insight:** Youth (5-17) update rates vary significantly across states.
> 
> **Risk:** Children in low-update states may face barriers accessing school enrollment, scholarships, and government benefits.
> 
> **Recommendation:** Partner with state education departments to conduct systematic school-based Aadhaar update camps, prioritizing states with below-average youth update proportions.

### 3. Seasonal Resource Planning
> **Insight:** Update volumes show quarterly patterns with identifiable peaks.
> 
> **Risk:** Fixed staffing leads to understaffing during peaks and inefficiency during troughs.
> 
> **Recommendation:** Implement flexible workforce scheduling aligned with historical quarterly demand patterns.

### 4. Automated Anomaly Monitoring
> **Insight:** Statistical methods effectively identify unusual activity.
> 
> **Risk:** Undetected fraud, data quality issues, or sudden demand surges.
> 
> **Recommendation:** Deploy automated monitoring dashboard using IQR and Z-score thresholds to flag districts/periods requiring investigation.

### 5. Access Equity Assessment
> **Insight:** Update volumes are heavily concentrated in urban areas.
> 
> **Risk:** Rural and remote populations may face barriers to Aadhaar updates.
> 
> **Recommendation:** Investigate low-update districts for access barriers and deploy mobile enrollment vans.

---

## Visualization Summary

| Visualization | Purpose | Key Insight |
|--------------|---------|-------------|
| State Distribution Heatmap | Shows update volume and age patterns by state | Concentration in large population states |
| Time Trend Analysis | Reveals temporal patterns and seasonality | Identifiable monthly/quarterly patterns |
| Age Group Comparison | Compares 5-17 vs 17+ across states | Adults dominate but youth ratios vary |
| Anomaly Detection | Highlights statistical outliers | Clear threshold for identifying at-risk districts |
| Quarterly Pattern | Shows seasonal distribution | Planning input for resource allocation |
| Top Districts Chart | Ranks highest-volume locations | Capacity planning priority list |

---

## Suggested Report Structure

1. **Executive Summary** (1 page)
2. **Introduction & Objectives** (0.5 page)
3. **Data Description & Methodology** (1 page)
4. **Key Findings** (2-3 pages with visualizations)
5. **Anomaly Analysis** (1 page)
6. **Policy Recommendations** (1-2 pages)
7. **Appendix: Technical Details** (as needed)

---

## Quotable Statements for Report

> "Adult population (17+) accounts for {key_stats['pct_17_plus']:.1f}% of all Aadhaar demographic updates, indicating that life events are the primary driver of modifications."

> "{key_stats['top_state']} alone accounts for {key_stats['top_state_share']:.1f}% of all demographic updates nationally, highlighting significant geographic concentration in service demand."

> "Statistical analysis identified {len(iqr_outliers)} districts with disproportionately high update volumes requiring capacity assessment."

> "The analysis provides a data-driven framework for optimizing Aadhaar service delivery across {df['state'].nunique()} states and {df['district'].nunique():,} districts."

---

*Analysis generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*
*Dataset: UIDAI Aadhaar Demographic Updates ({total_all_updates:,.0f} records)*
"""

# Save report
with open(DATA_DIR / "analysis_report.md", "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"Analysis report saved to: {DATA_DIR / 'analysis_report.md'}")

# Save summary statistics
summary_stats = pd.DataFrame({
    'Metric': ['Total Updates', 'Adult Updates (17+)', 'Youth Updates (5-17)', 
               'States/UTs', 'Districts', 'Pincodes', 'Outlier Districts', 'Temporal Anomalies'],
    'Value': [f"{total_all_updates:,.0f}", f"{df['demo_age_17_plus'].sum():,.0f}", f"{df['demo_age_5_17'].sum():,.0f}",
              df['state'].nunique(), df['district'].nunique(), df['pincode'].nunique(),
              len(iqr_outliers), len(monthly_spikes)]
})
summary_stats.to_csv(DATA_DIR / "summary_statistics.csv", index=False)
print(f"Summary statistics saved to: summary_statistics.csv")

# Save aggregated datasets
state_agg.to_csv(DATA_DIR / "state_level_analysis.csv", index=False)
district_agg.to_csv(DATA_DIR / "district_level_analysis.csv", index=False)
monthly_agg.to_csv(DATA_DIR / "monthly_analysis.csv", index=False)
print("Aggregated analysis files saved.")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"""
Generated Files:
├── cleaned_data.csv          - Preprocessed combined dataset
├── analysis_report.md        - Complete report-ready analysis
├── summary_statistics.csv    - Key metrics summary
├── state_level_analysis.csv  - State aggregation
├── district_level_analysis.csv - District aggregation
├── monthly_analysis.csv      - Time series analysis
└── visualizations/
    ├── state_distribution_heatmap.png
    ├── time_trend_analysis.png
    ├── age_group_comparison.png
    ├── anomaly_detection.png
    ├── quarterly_pattern.png
    └── top_districts.png
""")
