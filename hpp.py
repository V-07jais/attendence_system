import pandas as pd
import joblib

# 1. Load the complete saved pipeline
pipeline = joblib.load("house_rent_pipeline.pkl")

# 2. Define your new house data (using the exact capitalization from your dataset)
new_house = pd.DataFrame({
    "BHK": [2],
    "Size": [1200],
    "Area Type": ["Super Area"],
    "City": ["Delhi"],
    "Furnishing Status": ["Semi-Furnished"],
    "Tenant Preferred": ["Family"],
    "Bathroom": [2],
    "Point of Contact": ["Contact Owner"]
})

# 3. Predict directly using the pipeline
prediction = pipeline.predict(new_house)

# 4. Print the clean output
print(f"Predicted Rent: {prediction[0]:,.0f}")