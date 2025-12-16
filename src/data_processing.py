import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import os
import category_encoders as ce

def load_raw_data(path: str = "data/raw/data.csv") -> pd.DataFrame:
    """Load raw data from CSV."""
    if not os.path.exists(path):
        # Fallback if running from src
        if os.path.exists(os.path.join("..", path)):
             path = os.path.join("..", path)
    
    df = pd.read_csv(path, parse_dates=['TransactionStartTime'])
    return df

def calculate_iv(df, feature, target):
    """Calculate Information Value (IV) for a feature."""
    lst = []
    
    # Check if numeric; if so, bin it. If object, use as is.
    if np.issubdtype(df[feature].dtype, np.number) and df[feature].nunique() > 10:
        # Simple quantile binning for numeric high cardinality
        try:
            df[feature] = pd.qcut(df[feature], q=10, duplicates='drop').astype(str)
        except Exception:
            pass # Keep as is if qcut fails
            
    for val in df[feature].unique():
        all_col = df[df[feature] == val].count()[feature]
        good_col = df[(df[feature] == val) & (df[target] == 0)].count()[feature]
        bad_col = df[(df[feature] == val) & (df[target] == 1)].count()[feature]
        
        lst.append({
            'Value': val,
            'All': all_col,
            'Good': good_col,
            'Bad': bad_col
        })
        
    data = pd.DataFrame(lst)
    data['Distr_Good'] = data['Good'] / data['Good'].sum()
    data['Distr_Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distr_Good'] / data['Distr_Bad'])
    data.replace([np.inf, -np.inf], 0, inplace=True)
    data['IV'] = (data['Distr_Good'] - data['Distr_Bad']) * data['WoE']
    
    iv = data['IV'].sum()
    return iv

def apply_woe_transformation(df: pd.DataFrame, target_col: str, exclude_cols: list) -> pd.DataFrame:
    """Apply WoE transformation using category_encoders."""
    # Select potential features for WoE
    features_to_woe = [col for col in df.columns if col not in exclude_cols and col != target_col]
    
    # Focus on object columns
    cat_features = [col for col in features_to_woe if df[col].dtype == 'object']
    
    if not cat_features:
        return df

    print("Calculating WoE and IV...")
    
    # Calculate IV for each feature before transformation (for reporting)
    iv_stats = []
    for col in cat_features:
        try:
            iv = calculate_iv(df[[col, target_col]].copy(), col, target_col)
            iv_stats.append({'Feature': col, 'IV': iv})
        except Exception as e:
            print(f"Could not calc IV for {col}: {e}")

    if iv_stats:
        print("\nInformation Value (IV) Stats:")
        print(pd.DataFrame(iv_stats))

    # Apply transformation
    encoder = ce.WOEEncoder(cols=cat_features)
    # WOEEncoder requires numeric target. is_high_risk is int (0/1).
    try:
        encoded_df = encoder.fit_transform(df[cat_features], df[target_col])
        encoded_df.columns = [f"{col}_woe" for col in encoded_df.columns]
        
        # Concatenate
        df = pd.concat([df, encoded_df], axis=1)
    except Exception as e:
        print(f"WoE encoding failed: {e}")
        
    return df

def create_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregate features per customer."""
    agg = df.groupby('CustomerId').agg(
        total_transaction_amount=('Amount', 'sum'),
        avg_transaction_amount=('Amount', 'mean'),
        transaction_count=('TransactionId', 'count'),
        std_transaction_amount=('Amount', 'std'),
        avg_transaction_hour=('TransactionStartTime', lambda x: x.dt.hour.mean()),
        avg_transaction_day=('TransactionStartTime', lambda x: x.dt.day.mean()),
        avg_transaction_month=('TransactionStartTime', lambda x: x.dt.month.mean())
    ).reset_index()
    agg['std_transaction_amount'] = agg['std_transaction_amount'].fillna(0)
    return agg

def create_most_common_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Extract most common categorical values per customer."""
    cat_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy', 'ProviderId', 'ProductId']
    most_common = df.groupby('CustomerId')[cat_cols].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').reset_index()
    most_common.columns = ['CustomerId'] + [f'most_common_{col}' for col in cat_cols]
    return most_common

def calculate_rfm(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """Calculate RFM metrics."""
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max()
    
    rfm = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', lambda x: x.abs().sum())
    ).reset_index()
    return rfm

def create_proxy_target(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Create proxy target variable 'is_high_risk' using KMeans clustering on RFM."""
    # Scale RFM
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    
    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify high-risk cluster (lowest Frequency and Monetary, highest Recency usually)
    # We will assume high risk = "bad" = low engagement (low freq, low monetary)
    cluster_means = rfm_df.groupby('cluster')[['Frequency', 'Monetary']].mean()
    # sum of freq and monetary to find the "best" cluster, so the "worst" is the min
    # Or strict definition: High risk = those who might default. 
    # In buy-now-pay-later, maybe "disengaged" users are NOT high risk (they just don't use it). 
    # But the prompt says: "programmatically identify 'disengaged' customers and label them as high-risk proxies."
    # Wait, usually disengaged means we don't know enough, or they churned. 
    # "High-risk groups are those with a high likelihood of default... identify 'disengaged' customers and label them as high-risk proxies"
    # Okay, I will follow the prompt: Disengaged = High Risk.
    
    high_risk_cluster = cluster_means.mean(axis=1).idxmin()
    
    rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)
    rfm_df.drop('cluster', axis=1, inplace=True)
    return rfm_df


def feature_engineering_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Execute the full feature engineering and labeling pipeline."""
    
    # 1. Aggregate Features
    agg_df = create_aggregates(raw_df)
    
    # 2. Extract Categoricals
    cat_df = create_most_common_categoricals(raw_df)
    
    # 3. RFM
    rfm_df = calculate_rfm(raw_df)
    
    # 4. Proxy Target
    rfm_with_target = create_proxy_target(rfm_df)
    
    # Merge all
    processed = agg_df.merge(cat_df, on='CustomerId').merge(rfm_with_target, on='CustomerId')
    
    # 5. WoE Transformation
    # We need to be careful not to leak info if we were doing this inside CV, 
    # but for "feature engineering" task output, we often do it on the whole dataset 
    # or just provide the script to do it. 
    # The prompt says "Feature Engineering with WoE and IV".
    processed = apply_woe_transformation(processed, 'is_high_risk', exclude_cols=['CustomerId'])
    
    # 6. Scaling Numerical Features (excluding target and ID and WoE cols if we want to keep them raw)
    # Typically we scale raw numericals.
    num_cols = [
        'total_transaction_amount', 'avg_transaction_amount', 'transaction_count',
        'std_transaction_amount', 'avg_transaction_hour', 'avg_transaction_day',
        'avg_transaction_month', 'Recency', 'Frequency', 'Monetary'
    ]
    
    scaler = StandardScaler()
    processed[num_cols] = scaler.fit_transform(processed[num_cols])
    
    # Handle missing values if any remain (though we utilized fillna/imputation logic in aggregates)
    processed.fillna(0, inplace=True)
    
    return processed

if __name__ == "__main__":
    # Example usage
    try:
        df = load_raw_data()
        processed_df = feature_engineering_pipeline(df)
        
        output_path = "data/processed/processed_data.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_df.to_csv(output_path, index=False)
        print(f"Data processed and saved to {output_path}")
        print(processed_df.head())
    except Exception as e:
        print(f"Error during processing: {e}")