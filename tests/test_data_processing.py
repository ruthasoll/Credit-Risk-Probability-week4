# tests/test_data_processing.py

import pytest
import pandas as pd
import numpy as np
from src.data_processing import create_aggregates, calculate_rfm

@pytest.fixture
def sample_data():
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4'],
        'BatchId': ['B1', 'B1', 'B2', 'B2'],
        'AccountId': ['A1', 'A1', 'A2', 'A2'],
        'SubscriptionId': ['S1', 'S1', 'S2', 'S2'],
        'CustomerId': ['C1', 'C1', 'C2', 'C2'],
        'CurrencyCode': ['UGX', 'UGX', 'UGX', 'UGX'],
        'CountryCode': [256, 256, 256, 256],
        'ProviderId': ['P1', 'P1', 'P2', 'P2'],
        'ProductId': ['Prod1', 'Prod1', 'Prod2', 'Prod2'],
        'ProductCategory': ['Cat1', 'Cat1', 'Cat2', 'Cat2'],
        'ChannelId': ['Ch1', 'Ch1', 'Ch2', 'Ch2'],
        'Amount': [1000, 500, 2000, -100],
        'Value': [1000, 500, 2000, 100],
        'TransactionStartTime': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-05 12:00:00', '2023-01-06 13:00:00']),
        'PricingStrategy': [1, 1, 2, 2],
        'FraudResult': [0, 0, 0, 0]
    }
    return pd.DataFrame(data)

def test_create_aggregates(sample_data):
    agg = create_aggregates(sample_data)
    assert len(agg) == 2
    assert agg.loc[agg['CustomerId'] == 'C1', 'total_transaction_amount'].values[0] == 1500
    assert agg.loc[agg['CustomerId'] == 'C2', 'transaction_count'].values[0] == 2

def test_calculate_rfm(sample_data):
    # Snapshot date after the last transaction
    snapshot = pd.to_datetime('2023-01-10')
    rfm = calculate_rfm(sample_data, snapshot_date=snapshot)
    
    c1_recency = (snapshot - pd.to_datetime('2023-01-02 11:00:00')).days
    assert rfm.loc[rfm['CustomerId'] == 'C1', 'Recency'].values[0] == c1_recency
    
    assert rfm.loc[rfm['CustomerId'] == 'C2', 'Frequency'].values[0] == 2
    assert rfm.loc[rfm['CustomerId'] == 'C2', 'Monetary'].values[0] == 2100 # abs(2000) + abs(-100)
