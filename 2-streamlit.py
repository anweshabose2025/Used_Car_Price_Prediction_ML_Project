# (D:\Udemy\Complete_DSMLDLNLP_Bootcamp\UPractice1\venv) 
# D:\Udemy\Complete_DSMLDLNLP_Bootcamp\UPractice2\Used_Car_Price_Prediction>streamlit run 2-streamlit.py

import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="🚗 Used Car Price Prediction", layout="centered")

st.title("🚗 Used Car Price Prediction App")
st.markdown(
    """
    ### 🏁 Welcome!
    Curious about your car's **market value** today?  
    Fill in the details below, and this AI-powered app will estimate your car's **current selling price**.
    """)

st.divider()

with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

st.subheader("🔧 Enter Your Car Details")

col1, col2 = st.columns(2)

with col1:
    car_name = st.text_input('🚘 Car Name')
    brand = st.text_input('🏷️ Brand')
    selected_model = st.selectbox('🧩 Model', options=list(label_encoder.classes_))
    vehicle_age = st.number_input('⏳ Vehicle Age (Years)', min_value=0, max_value=50, step=1)
    km_driven = st.number_input('🛣️ Kilometers Driven', min_value=0, step=1000)
    seller_type = st.selectbox('🧍 Seller Type', options=['Trustmark Dealer', 'Individual', 'Dealer'])

with col2:
    fuel_type = st.selectbox('⛽ Fuel Type', options=['Petrol','Diesel','CNG','LPG','Electric'])
    transmission_type = st.selectbox('⚙️ Transmission', options=['Manual', 'Automatic'])
    mileage = st.number_input('📏 Mileage (km/l)', min_value=0.0, step=0.1)
    engine = st.selectbox('🔩 Engine Size (cc)', options=[
        796, 1197, 998, 1498, 1582, 1461, 1998, 1248, 2143, 2494, 1598,
        1497, 2523, 2982, 999, 1199, 1086, 2179, 1396, 1198, 1590, 1997,
        1493, 1120, 1196, 1186, 1061, 799, 1796, 2993, 1298, 3198, 1995,
        1373, 1591, 1968, 1399, 1047, 1353, 2755, 1395, 2498, 1999, 2953,
        2393, 2967, 2998, 2995, 1956, 1462, 1799, 2499, 1451, 1194, 1798,
        2694, 1496, 1969, 2487, 3604, 1499, 2987, 1599, 2354, 1950, 2148,
        4134, 2400, 1984, 2979, 2609, 2092, 5998, 1330, 2956, 3598, 2362,
        793, 1368, 3597, 1991, 5461, 1343, 2497, 4367, 1797, 3498, 2698,
        1896, 2999, 3855, 4806, 1985, 2771, 2157, 3628, 2198, 2996, 2199,
        4395, 3456, 4663, 2773, 6592, 2495, 1781, 2997, 1597, 2596, 4163
    ])
    max_power = st.number_input('⚡ Max Power (bhp)', min_value=0.0, step=1.0)
    seats = st.number_input('🪑 Number of Seats', min_value=2, max_value=10, step=1)

st.divider()

if st.button("💰 Predict Selling Price"):

    vehicle_age = int(vehicle_age)
    km_driven = int(km_driven)
    mileage = float(mileage)
    max_power = float(max_power)
    seats = int(seats)
    engine = int(engine)

    df = pd.DataFrame({
        'car_name':[car_name],
        'brand':[brand],
        'model':[selected_model],
        'vehicle_age':[vehicle_age],
        'km_driven':[km_driven],
        'seller_type':[seller_type],
        'fuel_type':[fuel_type],
        'transmission_type':[transmission_type],
        'mileage':[mileage],
        'engine':[engine],
        'max_power':[max_power],
        'seats':[seats]
    })
    st.write(df)
    # Drop unused
    df.drop(columns=['car_name', 'brand'], axis=1, inplace=True)

    # Encode categorical values
    df['seller_type'] = df['seller_type'].map({'Trustmark Dealer':0, 'Individual':1, 'Dealer':2})
    df['transmission_type'] = df['transmission_type'].map({'Manual':0, 'Automatic':1})
    df['fuel_type'] = df['fuel_type'].map({'Petrol':4,'Diesel':3,'CNG':2,'LPG':1,'Electric':0})
    df['model'] = label_encoder.transform(df['model'])

    # Scale & Predict
    df = scaler.transform(df)
    prediction = model.predict(df)

    st.success(f"✅ The predicted Used-Car Price is: {prediction[0]}")

st.divider()
st.caption("🔍 Developed by Anwesha Bose | Data Science & Machine Learning Project")