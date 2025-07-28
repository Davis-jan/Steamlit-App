# Interactive Debt Forecasting App
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import shap
import pickle
import datetime
import base64
# from PIL import Image # Commented out due to potential FileNotFoundError

# Set page config
st.set_page_config(page_title="Debt Forecasting App", layout="wide")

# Background image using base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("data:image/jpg;base64,{encoded}");
             background-size: cover;
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


# Dark mode toggle
dark_mode = st.toggle("🌙 Dark Mode")
if dark_mode:
    st.markdown(
        """
        <style>
        body { background-color: #1e1e1e; color: white; }
        .stApp {{ color: white; }}
        </style>
        """, unsafe_allow_html=True
    )

# App Title
st.title("📊 Debt Forecasting App")
st.subheader("Predict future public debt using Decision Tree Regression")


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Admin\Downloads\42346012_Public Debt (2).csv")
    # Rename Columns
    new_column_names = {
        df.columns[0]: 'Year',
        df.columns[1]: 'Month',
        df.columns[2]: 'Domestic Debt',
        df.columns[3]: 'External Debt',
        df.columns[4]: 'Total'
    }
    df = df.rename(columns=new_column_names)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Preprocess Columns
    for col in ['Domestic Debt', 'External Debt', 'Total']:
        df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
        # Use errors='coerce' to handle non-numeric values
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values created by coercion
    df.dropna(subset=['Domestic Debt', 'External Debt', 'Total'], inplace=True)

    # Convert Date
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    df = df.set_index('Date')

    return df

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio("Menu", ["📊 Dashboard", "📈 Prediction", "📥 Download Data"])

# Main title
st.markdown("<h1 style='text-align: center;'>Debt Forecasting App</h1>", unsafe_allow_html=True)
st.markdown("**Dataset: Kenya Public Debt Over Time**")


# Section: Dashboard
if section == "📊 Dashboard":
    st.subheader("Debt Trend Overview")

    # Dropdown filter
    debt_types = ['Domestic Debt', 'External Debt', 'Total'] # Use the cleaned column names
    selected_type = st.selectbox("Select Debt Type", debt_types)

    # Line Chart
    fig = px.line(df, x=df.index, y=selected_type, title=f"{selected_type} Over the Years", markers=True) # Use index for date
    st.plotly_chart(fig, use_container_width=True)

    # Show raw data
    with st.expander("🔍 View Raw Data"):
        st.dataframe(df)

# Section: Prediction
elif section == "📈 Prediction":
    st.subheader("Predict Future Debt")

    # Feature and Target
    df_pred = df.copy()
    # No need to dropna again as it's done in load_data

    features = ['Domestic Debt', 'External Debt']
    target = 'Total'

    X = df_pred[features]
    y = df_pred[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Training
    model = DecisionTreeRegressor(random_state=42)
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    # Evaluation
    y_pred = best_model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write(f"Mean Absolute Error: {mae:.2f}")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")
    st.write("Best Parameters:", grid_search.best_params_)


    # Forecasting
    st.subheader("Make a Prediction")
    domestic_input = st.number_input("Enter Domestic Debt", min_value=0.0)
    external_input = st.number_input("Enter External Debt", min_value=0.0)

    if st.button("Predict Total Debt"):
        input_data = scaler.transform([[domestic_input, external_input]])
        prediction = best_model.predict(input_data)[0]
        st.success(f"Predicted Total Debt: {prediction:,.2f}")

    # Save Model and Scaler
    with open('decision_tree_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


# Section: Download Data
elif section == "📥 Download Data":
    st.subheader("Download Processed Data")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='processed_debt_data.csv',
        mime='text/csv'
    )

# Debt Trend
st.subheader("Total Debt Over Time")
fig1, ax1 = plt.subplots()
ax1.plot(df.index, df['Total'], label='Total Debt')
ax1.set_xlabel("Date")
ax1.set_ylabel("Debt")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)
plt.close(fig1) # Added to close the figure



# Footer
st.markdown("---")
st.caption("Developed by Davies Ochieng Owuor")

