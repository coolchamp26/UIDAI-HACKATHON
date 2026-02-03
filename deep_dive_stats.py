
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

try:
    df_dist = pd.read_csv('district_level_analysis.csv')
    
    # Distribution Metrics
    dist_kurtosis = kurtosis(df_dist['total_updates'])
    dist_skew = skew(df_dist['total_updates'])
    dist_gini = gini(df_dist['total_updates'].values)
    
    # Correlations
    corr_pincode = df_dist['total_updates'].corr(df_dist['unique_pincodes'])
    corr_age = df_dist['updates_5_17'].corr(df_dist['updates_17_plus'])
    
    # Top 1% handling
    top_1_percent_count = int(len(df_dist) * 0.01)
    top_1_percent_vol = df_dist.nlargest(top_1_percent_count, 'total_updates')['total_updates'].sum()
    total_vol = df_dist['total_updates'].sum()
    concentration_ratio = (top_1_percent_vol / total_vol) * 100

    print(f"DIST_KURTOSIS:{dist_kurtosis:.4f}")
    print(f"DIST_SKEW:{dist_skew:.4f}")
    print(f"DIST_GINI:{dist_gini:.4f}")
    print(f"CORR_PINCODE_UPDATES:{corr_pincode:.4f}")
    print(f"CORR_YOUTH_ADULT:{corr_age:.4f}")
    print(f"CONCENTRATION_TOP_1_PCT:{concentration_ratio:.2f}")

    # Monthly Variance
    df_month = pd.read_csv('monthly_analysis.csv')
    month_cv = df_month['total_updates'].std() / df_month['total_updates'].mean()
    print(f"MONTHLY_CV:{month_cv:.4f}")

except Exception as e:
    print(f"Error: {e}")
