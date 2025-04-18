def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer



    # Step 1: Check for Missing Values
    missing_percentage = data_raw.isnull().mean() * 100
    columns_to_drop = missing_percentage[missing_percentage > 40].index
    data_cleaned = data_raw.drop(columns=columns_to_drop)
    
    # Log changes
    log_changes = f"Dropped columns with >40% missing values: {list(columns_to_drop)}"
    
    # Step 2: Impute Missing Values
    numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns
    categorical_cols = data_cleaned.select_dtypes(include=['object']).columns
    
    # Impute numeric columns with mean
    num_imputer = SimpleImputer(strategy='mean')
    data_cleaned[numeric_cols] = num_imputer.fit_transform(data_cleaned[numeric_cols])
    
    # Impute categorical columns with mode
    cat_imputer = SimpleImputer(strategy='most_frequent')
    data_cleaned[categorical_cols] = cat_imputer.fit_transform(data_cleaned[categorical_cols])
    
    # Step 3: Convert Data Types
    data_cleaned['TotalCharges'] = pd.to_numeric(data_cleaned['TotalCharges'], errors='coerce')
    
    # Step 4: Remove Duplicate Rows
    initial_row_count = data_cleaned.shape[0]
    data_cleaned = data_cleaned.drop_duplicates()
    log_changes += f"\nRemoved {initial_row_count - data_cleaned.shape[0]} duplicate rows."
    
    # Step 5: Remove Rows with Missing Values
    initial_row_count = data_cleaned.shape[0]
    data_cleaned = data_cleaned.dropna()
    log_changes += f"\nRemoved {initial_row_count - data_cleaned.shape[0]} rows with remaining missing values."
    
    # Step 6: Analyze for Extreme Outliers
    for col in numeric_cols:
        Q1 = data_cleaned[col].quantile(0.25)
        Q3 = data_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        outliers = data_cleaned[(data_cleaned[col] < lower_bound) | (data_cleaned[col] > upper_bound)]
        if not outliers.empty:
            log_changes += f"\nExtreme outliers detected in column '{col}': {outliers.shape[0]} rows."
    
    # Step 7: Final Review
    final_row_count = data_cleaned.shape[0]
    log_changes += f"\nFinal dataset contains {final_row_count} rows."
    
    # Step 8: Document Changes
    print(log_changes)
    
    return data_cleaned