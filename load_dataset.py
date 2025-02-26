import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

###############################
# Adult Dataset Functions
###############################

def preprocess_adult(df):
    """
    Preprocesses the Adult dataset.
    
    Selected features (to keep dimensionality low):
      - Numeric: age, education-num, hours-per-week
      - Categorical/Binary: marital-status, sex
      - Target: income (converted to binary)
    
    Steps:
      1. Replace "?" with np.nan.
      2. Subset to the chosen columns.
      3. Rename columns to use underscores instead of hyphens.
      4. Impute missing values (median for numeric, mode for categorical).
      5. Map 'sex' to binary (Male -> 1, Female -> 0).
      6. Convert 'income' to binary (">50K" -> 1, "<=50K" -> 0) and cast to int.
      7. One-hot encode 'marital_status' (drop the first category) and convert dummy columns to int.
      8. Scale numeric features using StandardScaler.
    
    Returns:
        X (pd.DataFrame): Transformed features.
        y (pd.Series): Binary target variable.
    """
    # Replace "?" with NaN
    df = df.replace("?", np.nan)
    
    # Keep only the selected columns
    df = df[["age", "education-num", "hours-per-week", "marital-status", "sex", "income"]]
    
    # Rename columns: replace hyphens with underscores for consistency.
    df.columns = df.columns.str.replace('-', '_')
    
    # Define columns based on the new names
    num_cols = ["age", "education_num", "hours_per_week"]
    cat_cols = ["marital_status", "sex", "income"]
    
    # Impute numeric columns with median
    num_imputer = SimpleImputer(strategy="median")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    # Impute categorical columns with the most frequent value
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    # Map binary feature 'sex' to numeric (assuming values "Male" and "Female")
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    
    # Clean and convert target 'income' to binary, then cast to int to avoid FutureWarning
    df["income"] = df["income"].str.strip().replace({">50K": '1', "<=50K": '0'}).astype(int)
    
    # One-hot encode 'marital_status'
    df = pd.get_dummies(df, columns=["marital_status"], drop_first=True)
    
    # Ensure one-hot encoded columns are of integer type
    dummy_cols = [col for col in df.columns if col.startswith("marital_status_")]
    df[dummy_cols] = df[dummy_cols].astype(int)
    
    # Scale numeric features
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Separate target variable
    y = df["income"]
    X = df.drop("income", axis=1)
    
    return X, y

def load_adult():
    """
    Loads and preprocesses the Adult dataset from 'data/adult/adult.data'.
    
    Returns:
        X (pd.DataFrame): Transformed feature set.
        y (pd.Series): Binary target variable.
    """
    # Define column names for the full Adult dataset
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income"
    ]
    
    # Read the data (the file has no header row)
    df = pd.read_csv("data/adult/adult.data", header=None, names=columns, skipinitialspace=True)
    
    # Preprocess and reduce features
    X, y = preprocess_adult(df)
    
    return X, y

###############################
# Housing Dataset Functions
###############################

def preprocess_housing(df):
    """
    Preprocesses the Boston Housing dataset.
    
    The dataset (after renaming) has the following columns:
      - Features: crime_rate, large_lot_residential_pct, industrial_area_pct,
                  near_river, nitric_oxide_level, avg_rooms, old_home_pct,
                  distance_to_employment, highway_access_index, property_tax_rate,
                  student_teacher_ratio, racial_index, lower_status_pct
      - Target: median_home_value
                  
    Steps:
      1. Impute missing values for all features (using median imputation).
      2. Scale all feature columns using StandardScaler.
    
    Returns:
        X (pd.DataFrame): Transformed features.
        y (pd.Series): Target variable.
    """
    # Identify feature columns (all except the target)
    feature_cols = [col for col in df.columns if col != "median_home_value"]
    
    # Impute missing values for numeric features
    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    
    # Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Separate target variable
    y = df["median_home_value"]
    X = df.drop("median_home_value", axis=1)
    
    return X, y

def load_housing():
    """
    Loads and preprocesses the Boston Housing dataset from 'data/housing.csv'.
    
    The CSV is assumed to be whitespace-delimited. The original columns are:
        CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
    which are then renamed to more descriptive names.
    
    Returns:
        X (pd.DataFrame): Transformed features.
        y (pd.Series): Target variable.
    """
    # Original column names
    original_columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
        "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B",
        "LSTAT", "MEDV"
    ]
    # New descriptive column names
    new_columns = [
        "crime_rate", "large_lot_residential_pct", "industrial_area_pct", "near_river",
        "nitric_oxide_level", "avg_rooms", "old_home_pct", "distance_to_employment",
        "highway_access_index", "property_tax_rate", "student_teacher_ratio",
        "racial_index", "lower_status_pct", "median_home_value"
    ]
    
    # Load the CSV with whitespace as the delimiter; assume no header row.
    df = pd.read_csv("data/housing.csv", sep=r"\s+", header=None, names=original_columns)
    
    # Rename columns to more descriptive names
    df.columns = new_columns
    
    # Preprocess the housing data
    X, y = preprocess_housing(df)
    
    return X, y
