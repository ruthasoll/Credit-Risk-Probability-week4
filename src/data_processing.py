# src/data_processing.py

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import category_encoders as ce
import joblib
import os

def load_raw_data(path: str = "../data/raw/train.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['TransactionStartTime'])
    return df

def create_features_and_proxy(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """Main function: full feature engineering + RFM + proxy target"""
    
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max()
    
    # 1. Aggregations
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
    
    # 2. Most common categoricals
    cat_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy', 'ProviderId', 'ProductId']
    for col in cat_cols:
        mode_series = df.groupby('CustomerId')[col].apply(
            lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
        )
        agg[f'most_common_{col}'] = agg['CustomerId'].map(mode_series)
    
    # 3. Label encode high cardinality
    le = LabelEncoder()
    for col in ['most_common_ProductCategory', 'most_common_ProductId']:
        agg[f'encoded_{col.split("_")[2]}'] = le.fit_transform(agg[col].astype(str))
        agg.drop(col, axis=1, inplace=True)
    
    # 4. RFM
    rfm = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', lambda x: x.abs().sum())
    ).reset_index()
    
    processed = agg.merge(rfm[['CustomerId', 'Recency', 'Frequency', 'Monetary']], on='CustomerId')
    
    # 5. Scale RFM for clustering
    rfm_scaler = StandardScaler()
    rfm_scaled = rfm_scaler.fit_transform(processed[['Recency', 'Frequency', 'Monetary']])
    
    # 6. K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    processed['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # 7. Identify high-risk cluster (lowest frequency/monetary)
    cluster_means = processed.groupby('cluster')[['Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_means.mean(axis=1).idxmin()  # or use .idxmin() on Frequency
    
    processed['is_high_risk'] = (processed['cluster'] == high_risk_cluster).astype(int)
    processed.drop('cluster', axis=1, inplace=True)
    
    # 8. Final scaling of numerical features for modeling
    num_cols = ['total_transaction_amount', 'avg_transaction_amount', 'transaction_count',
                'std_transaction_amount', 'avg_transaction_hour', 'avg_transaction_day',
                'avg_transaction_month', 'Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    processed[num_cols] = scaler.fit_transform(processed[num_cols])
    
    return processed, {'rfm_scaler': rfm_scaler, 'kmeans': kmeans, 'num_scaler': scaler}

def save_processed_data(df: pd.DataFrame, path: str = "../data/processed/processed_with_target.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

# Optional: run when executing script directly
if __name__ == "__main__":
    df_raw = load_raw_data()
    processed_df, artifacts = create_features_and_proxy(df_raw)
    save_processed_data(processed_df)
    print("Processing complete. Saved to data/processed/")