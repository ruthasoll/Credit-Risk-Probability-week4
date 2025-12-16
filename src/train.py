from src.data_processing import create_features_and_proxy

processed_df, _ = create_features_and_proxy(df_raw)
X = processed_df.drop(['CustomerId', 'is_high_risk'], axis=1)
y = processed_df['is_high_risk']