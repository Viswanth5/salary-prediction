# frontend.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

def show_ui(model, predict_salary):
    st.set_page_config(page_title="Salary Predictor", layout="centered")
    st.title("💼 Salary Predictor App")
    st.write("This app predicts salary based on years of experience using a Linear Regression model.")

    # Sidebar
    st.sidebar.header("⚙ Options")
    uploaded_file = st.sidebar.file_uploader("Upload a custom CSV file", type=["csv"])

    # Load dataset
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("Custom dataset loaded successfully.")
    else:
        data = pd.read_csv("Salary_dataset.csv")
        st.info("Using default Salary_dataset.csv")

    # Display data
    with st.expander("📊 Preview Dataset"):
        st.dataframe(data)

    # Show scatter plot
    with st.expander("📈 Years of Experience vs Salary Chart"):
        fig, ax = plt.subplots()
        ax.scatter(data['YearsExperience'], data['Salary'], color='blue')
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary")
        ax.set_title("Experience vs Salary")
        st.pyplot(fig)

    # Input and prediction
    st.subheader("🔍 Predict Salary")
    years = st.number_input("Enter Years of Experience", min_value=0.0, step=0.1)

    if st.button("Predict Salary"):
        salary = predict_salary(model, years)
        st.success(f"🎯 Predicted Salary: ₹{salary:,.2f}")

        # Download result
        result_df = pd.DataFrame([[years, salary]], columns=["YearsExperience", "PredictedSalary"])
        buffer = BytesIO()
        result_df.to_csv(buffer, index=False)
        st.download_button("Download Prediction", buffer.getvalue(), file_name="salary_prediction.csv", mime="text/csv")
