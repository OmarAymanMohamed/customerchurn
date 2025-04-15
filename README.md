# Customer Churn Prediction App

This is a Streamlit web application that predicts customer churn probability based on various customer attributes and service usage patterns.

## Features

- Interactive web interface for inputting customer data
- Real-time churn probability prediction
- Visual representation of prediction results
- Risk assessment and recommendations
- Responsive design that works on both desktop and mobile devices

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the App Locally

1. Ensure you have all the model artifacts in the `model_artifacts` directory:
   - churn_model.joblib
   - scaler.joblib
   - feature_names.txt

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your web browser and go to `http://localhost:8501`

## Deploying to Streamlit Cloud

1. Create a free account on [Streamlit Cloud](https://streamlit.io/cloud)

2. Connect your GitHub repository to Streamlit Cloud

3. Deploy the app by selecting the repository and branch

4. Your app will be available at a public URL

## Model Information

The app uses a machine learning model trained on customer churn data with the following features:
- Customer demographics (gender, senior citizen status, etc.)
- Service subscriptions (phone, internet, etc.)
- Contract details
- Billing information
- Service usage patterns

## Contributing

Feel free to submit issues and enhancement requests! 