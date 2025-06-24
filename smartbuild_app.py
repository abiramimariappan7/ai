import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="SmartBuild", layout="centered")
st.title("🏗️ SmartBuild – Estimator & Price Predictor")

st.sidebar.header("Select Tool")
tool = st.sidebar.radio("Choose Function", ["Block Estimator", "Price Predictor"])

# --- BLOCK ESTIMATOR ---
def calculate_blocks_needed(length_ft, width_ft, height_ft, wall_thickness_mm,
                            block_length_mm, block_height_mm, block_width_mm,
                            num_doors=1, num_windows=2,
                            door_area=21, window_area=12):

    length_m = length_ft * 0.3048
    width_m = width_ft * 0.3048
    height_m = height_ft * 0.3048
    wall_thickness_m = wall_thickness_mm / 1000

    total_wall_area = 2 * (length_m + width_m) * height_m
    opening_area = (door_area * num_doors + window_area * num_windows) * 0.0929
    net_wall_area = total_wall_area - opening_area

    wall_volume = net_wall_area * wall_thickness_m
    block_volume = (block_length_mm / 1000) * (block_width_mm / 1000) * (block_height_mm / 1000)

    total_blocks = wall_volume / block_volume
    total_blocks_with_wastage = round(total_blocks * 1.1)

    return total_blocks_with_wastage

# --- PRICE PREDICTION MODEL ---
def train_price_model():
    data = pd.DataFrame({
        "Demand_Level_Num": np.random.randint(0, 3, 200),
        "Blocks_Sold": np.random.randint(10000, 60000, 200),
        "Competitor_Price_per_Block": np.random.uniform(38, 50, 200),
        "Region_Bangalore": np.random.randint(0, 2, 200),
        "Region_Chennai": np.random.randint(0, 2, 200),
        "Region_Coimbatore": np.random.randint(0, 2, 200),
        "Region_Hyderabad": np.random.randint(0, 2, 200),
        "Region_Pune": np.random.randint(0, 2, 200),
        "Month": np.random.randint(1, 13, 200),
        "MEP_Price_per_Block": np.random.uniform(38, 50, 200)
    })

    X = data.drop("MEP_Price_per_Block", axis=1)
    y = data["MEP_Price_per_Block"]

    model = RandomForestRegressor().fit(X, y)
    return model, X.columns

# --- UI Logic ---
if tool == "Block Estimator":
    st.subheader("🧱 AAC Block Estimator")

    length = st.number_input("Room Length (ft)", 5.0, 100.0, 10.0)
    width = st.number_input("Room Width (ft)", 5.0, 100.0, 10.0)
    height = st.number_input("Room Height (ft)", 6.0, 15.0, 10.0)

    wall_thickness = st.number_input("Wall Thickness (mm)", 75, 300, 150)
    block_l = st.number_input("Block Length (mm)", 400, 800, 600)
    block_w = st.number_input("Block Width (mm)", 100, 300, 150)
    block_h = st.number_input("Block Height (mm)", 150, 300, 200)

    doors = st.slider("Number of Doors", 0, 5, 1)
    windows = st.slider("Number of Windows", 0, 5, 2)

    if st.button("Calculate Blocks Needed"):
        total_blocks = calculate_blocks_needed(length, width, height,
                                               wall_thickness,
                                               block_l, block_h, block_w,
                                               doors, windows)
        st.success(f"Estimated Blocks Needed: {total_blocks}")

else:
    st.subheader("💰 Block Price Predictor")

    region = st.selectbox("Region", ["Chennai", "Bangalore", "Hyderabad", "Pune", "Coimbatore"])
    demand_level = st.selectbox("Demand Level", ["Low", "Medium", "High"])
    blocks_sold = st.number_input("Blocks Sold", 5000, 100000, 50000)
    competitor_price = st.number_input("Competitor Price (₹)", 35.0, 60.0, 45.0)
    today = date.today()
    month = st.slider("Month", 1, 12, today.month)

    if st.button("Predict Price"):
        model, feature_cols = train_price_model()

        input_data = {
            'Demand_Level_Num': {"Low": 0, "Medium": 1, "High": 2}[demand_level],
            'Blocks_Sold': blocks_sold,
            'Competitor_Price_per_Block': competitor_price,
            'Region_Bangalore': 1 if region == "Bangalore" else 0,
            'Region_Chennai': 1 if region == "Chennai" else 0,
            'Region_Coimbatore': 1 if region == "Coimbatore" else 0,
            'Region_Hyderabad': 1 if region == "Hyderabad" else 0,
            'Region_Pune': 1 if region == "Pune" else 0,
            'Month': month
        }

        input_df = pd.DataFrame([input_data])[feature_cols]
        predicted = model.predict(input_df)[0]
        st.success(f"📦 Predicted MEP Selling Price: ₹{predicted:.2f} per block")
