import pandas as pd
from joblib import load

def data_gathering():
    # New data input
    print("Please enter the following customer information:")
    data = {
        'Age': float(input("Age: ")),
        'Gender': input("Gender (Male/Female): "),
        'Support Calls': float(input("Number of Support Calls: ")),
        'Payment Delay': float(input("Payment Delay (days): ")),
        'Contract Length': input("Contract Length (Annual/Quarterly/Monthly): "),
        'Total Spend': float(input("Total Spend: ")),
        'Last Interaction': float(input("Days since Last Interaction: "))
    }
    return pd.DataFrame([data])

def transform_categorical(df):
    # One-hot encode categorical variables

    df_encoded = df.copy()

    print(df_encoded)
    
    # Building of new features
    df_encoded["Gender_Male"] = (df_encoded["Gender"] == "Male").astype(float)
    df_encoded["Contract Length_Monthly"] = (df_encoded["Contract Length"] == "Monthly").astype(float)
    df_encoded["Contract Length_Quarterly"] = (df_encoded["Contract Length"] == "Quarterly").astype(float)

    print(df_encoded)

    # Dropping previous feature
    df_encoded.drop(["Contract Length"], axis=1, inplace=True)

    # Reordering order og columns to fit the model correctly
    order_of_columns = ['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Gender_Male', 'Contract Length_Monthly', 'Contract Length_Quarterly']
    df_encoded = df_encoded[order_of_columns]

    print(df_encoded)

    print("Columns after dropping:", df_encoded.columns.tolist())
    return df_encoded

def predict_churn_pca(model, scaler, pca_model, data):
    # Function for prediction (including PCA)
    try:
        scaled_data = scaler.transform(data)
        pca_data = pca_model.transform(scaled_data)
        prediction = model.predict(pca_data)
        return prediction[0]
    except Exception as e:
        print(f"Error during PCA transformation or prediction: {e}")
        raise

def predict_churn(model, scaler, data):
    # Function for prediction
    try:
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)
        return prediction[0]
    except Exception as e:
        print(f"Error during scaling or prediction: {e}")
        raise

if __name__=="__main__":
    model = load("svm_churn_model2.joblib")
    scaler = load("scaler2.joblib")
    pca_model = load("pca_reduce2.joblib")

    while True:
        try:
            user_data = data_gathering()
            transformed_data = transform_categorical(user_data)
            
            # Ensure transformed_data is in the right shape for the scaler and model
            if transformed_data.shape[1] != scaler.n_features_in_:
                raise ValueError("The transformed data has an incorrect number of features.")
            
            prediction = predict_churn_pca(model, scaler, pca_model, transformed_data)
            
            print(f"\nChurn Prediction: {'Yes' if prediction == 1 else 'No'}")
            
        except ValueError as e:
            print(f"Value Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
        if input("\nMake another prediction? (y/n): ").lower() != 'y':
            break
