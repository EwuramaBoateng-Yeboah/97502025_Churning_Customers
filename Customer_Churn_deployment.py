
import pickle
import streamlit as st
import tensorflow as tf
import numpy as np
from keras.models import load_model


# Opens the deployed model
model=load_model('deploy.h5')
#scaling = pickle.load(open('scalar_model.pkl', 'rb'))
with open("scalar_model.pkl", "rb") as f:
    scaler= pickle.load(f)


def predict(customer_inputs):
    st.title('Customer Churn Prediction')

    # Convert input features to numpy array
    # customer_inputs = np.array(customer_inputs)

    # Print the shape and type of customer_inputs before transformation
    # print("Before transformation - Shape:", customer_inputs.shape, "Type:", customer_inputs.dtype)

    # # Reshape the input data to match the expected shape
    # customer_inputs = customer_inputs.reshape(1, -1)

    # # Print the shape and type of customer_inputs after reshaping
    # print("After reshaping - Shape:", customer_inputs.shape, "Type:", customer_inputs.dtype)



    # # Print the shape and type of scaled_inputs
    # print("After transformation - Shape:", scaled_inputs.shape, "Type:", scaled_inputs.dtype)

    # Make prediction
    makeprediction = model.predict([customer_inputs])
    confidence_factor = makeprediction[0][0] 

    return makeprediction,confidence_factor

def map_input_to_values(value, field):
    if field == 'InternetService':
        return {'DSL': 0, 'Fiber optic': 1, 'No': 2}.get(value)
    elif field == 'gender':
        return {'Female': 0, 'Male': 1}.get(value)
    elif field == 'PaymentMethod':
        return {'Electronic check': 2, 'Mailed check': 3, 'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1}.get(value)
    elif field == 'Contract':
        return {'Month-to-month': 0, 'One year': 1, 'Two year': 2}.get(value)
    elif field == 'TechSupport':
        return {'No': 0, 'Yes': 2, 'No internet service': 1}.get(value)
    elif field == 'OnlineBackup':
        return {'Yes': 2, 'No': 0, 'No internet service': 1}.get(value)
    elif field == 'OnlineSecurity':
        return {'No': 0, 'Yes': 2, 'No internet service': 1}.get(value)




    # Features
def main():
    # Input fields
    tenure = st.number_input('Tenure', value=0)
    MonthlyCharges = st.number_input('Monthly Charges', value=0.0)
    TotalCharges = st.number_input('Total Charges', value=0.0)

    Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    PaymentMethod = st.selectbox('PaymentMethod', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    TechSupport = st.selectbox('TechSupport', ['No', 'Yes', 'No internet service'])
    OnlineSecurity = st.selectbox('OnlineSecurity', ['No', 'Yes', 'No internet service'])
    gender = st.selectbox('Gender', ['Female', 'Male'])
    OnlineBackup = st.selectbox('OnlineBackup', ['Yes', 'No', 'No internet service'])

    # Map values to categorical features
    contract_value = map_input_to_values(Contract, 'Contract')
    payment_method_value = map_input_to_values(PaymentMethod, 'PaymentMethod')
    tech_support_value = map_input_to_values(TechSupport, 'TechSupport')
    online_security_value = map_input_to_values(OnlineSecurity, 'OnlineSecurity')
    gender_value = map_input_to_values(gender, 'gender')
    online_backup_value = map_input_to_values(OnlineBackup, 'OnlineBackup')
    

    
    if st.button('Predict', key='predict_button'):
            # Scale the input data using the loaded scaler
         # Combine the features into a 2D array
        
    # Ensure all variables are lists
        TotalCharges = [TotalCharges] if isinstance(TotalCharges, (int, float)) else TotalCharges
        MonthlyCharges = [MonthlyCharges] if isinstance(MonthlyCharges, (int, float)) else MonthlyCharges
        tenure = [tenure] if isinstance(tenure, (int, float)) else tenure
        contract_value = [contract_value] if isinstance(contract_value, (int, float)) else contract_value
        payment_method_value = [payment_method_value] if isinstance(payment_method_value, (int, float)) else payment_method_value
        tech_support_value = [tech_support_value] if isinstance(tech_support_value, (int, float)) else tech_support_value
        online_security_value = [online_security_value] if isinstance(online_security_value, (int, float)) else online_security_value
        gender_value = [gender_value] if isinstance(gender_value, (int, float)) else gender_value
        online_backup_value = [online_backup_value] if isinstance(online_backup_value, (int, float)) else online_backup_value


        # Combine the features into a 2D array
        data = list(zip(TotalCharges, MonthlyCharges, tenure,contract_value, payment_method_value, tech_support_value, online_security_value, gender_value, online_backup_value))

        numbers = scaler.transform(data)
        print(numbers)

        user_inputs = numbers[0]
        confidence_factor = predict(user_inputs)
        st.write(f'Confidence Factor: {confidence_factor}')


        TotalCharges=numbers[0][0]
        MonthlyCharges=numbers[0][1]
        tenure=numbers[0][2]
        contract_value=numbers[0][3]
        payment_method_value=numbers[0][4]
        tech_support_value=numbers[0][5]
        online_security_value=numbers[0][6]
        gender_value=numbers[0][7]
        online_backup_value=numbers[0][8]
        user_inputs = [TotalCharges,MonthlyCharges,tenure,contract_value, payment_method_value, tech_support_value, online_security_value, gender_value, online_backup_value]


        output = predict(user_inputs)
        if output[0][0]<0.5:
            output="No"
        else: output="Yes"
        st.success(f"The Predicted Customer Churn is {output}")


        


if __name__ == '__main__':
    main()


#go to terminal and type (streamlit run Customer_Churn_deployment.py)