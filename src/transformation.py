from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessing_pipeline():
    """
    Returns a production-grade transformation pipeline.
    """
    # 1. Select numeric features to scale (Dell logistics logic)
    # Scaling ensures 'Sales' (large numbers) don't dominate 'Days' (small numbers)
    numeric_features = ['Days_for_shipment', 'Sales', 'Order_Item_Quantity']
    
    # 2. Select categorical features to encode
    # These represent the 'Risk Factors' in shipping
    categorical_features = ['Type', 'Category_Name', 'Order_Region', 'Shipping_Mode', 'Customer_Segment']

    # 3. Define the Transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 4. Create the ColumnTransformer (The 'Brain' of our Feature Engineering)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Automatically drops any column we didn't specify (IDs, Names, etc.)
    )
    
    return preprocessor