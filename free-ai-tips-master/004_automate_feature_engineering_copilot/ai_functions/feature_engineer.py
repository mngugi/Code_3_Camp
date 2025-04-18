def feature_engineer(data_raw):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder



    # Step 1: Convert TotalCharges to float64
    data_raw['TotalCharges'] = pd.to_numeric(data_raw['TotalCharges'], errors='coerce')

    # Step 2: Remove unique string features
    data_raw = data_raw.drop(columns=['customerID'])

    # Step 3: Remove constant features (none identified in this dataset)

    # Step 4: Handle high cardinality categorical features (none identified in this dataset)

    # Step 5: One-Hot Encoding categorical features
    categorical_features = data_raw.select_dtypes(include=['object']).columns.tolist()
    categorical_features.remove('Churn')  # Exclude target variable

    one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
    one_hot_encoded = one_hot_encoder.fit_transform(data_raw[categorical_features])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_features))

    # Combine one-hot encoded features with the original dataframe (excluding original categorical columns)
    data_raw = data_raw.drop(columns=categorical_features)
    data_raw = pd.concat([data_raw.reset_index(drop=True), one_hot_encoded_df.reset_index(drop=True)], axis=1)

    # Step 6: Leave numeric features untransformed (they are already in the right format)

    # Step 7: No datetime columns to process

    # Step 8: Encode target variable
    label_encoder = LabelEncoder()
    data_raw['Churn'] = label_encoder.fit_transform(data_raw['Churn'])

    # Step 9: Convert Boolean to Integer (already handled in the dataset)

    # Step 10: No additional feature engineering steps required

    # Step 11: Final review
    data_engineered = data_raw

    return data_engineered