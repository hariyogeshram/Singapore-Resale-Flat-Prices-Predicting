# Import the Required Libraries

import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Streamlit page custom design

def streamlit_config():

   st.set_page_config(layout="wide")

   st.write("""
    <style>
        body {
            background-color: black;
        }
        .stApp {
            background-color: black;
        }
        h1 {
            font-size: 3.5em; /* Increase font size for header */
            color: yellow;
        }
        h2, h3, h4, h5, h6 {
            font-size: 3.5em;
            color: yellow; /* Ensure headers are visible */
        }
        p, li {
            font-size: 1.5em; /* Increase font size for paragraphs and list items */
            color: white;
        }
        .stButton > button {
            font-size: 2.8em; /* Increase font size for buttons */
        }
    </style>
    <div style='text-align:center'>
        <h1 style='color:#009999;'>Singapore Resale Flat Price Prediction</h1>
    </div>
    """, unsafe_allow_html=True)

# Custom style for submit button - color and width
def style_submit_button():

    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #367F89;
            color: black;
            width: 70%;
            margin: 20px auto;
            display: block;
        }
        </style>
    """, unsafe_allow_html=True)

# Custom style for prediction result text - color and position
def style_prediction():

    st.markdown("""
        <style>
        .center-text {
            text-align: center;
            color: #20CA0C;
        }
        </style>
    """, unsafe_allow_html=True)

# Load the model
with open("D://6_Singapore Resale Flat Prices Predicting//regression_model.pkl", 'rb') as f:
    model = pickle.load(f)

# Feature names
features = ['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model',
            'year', 'month_of_year', 'lease_commence_year',
            'remaining_lease_years', 'remaining_lease_months']

# Categorical variable mappings
categorical_mappings = {
    'town': {'SENGKANG': 20, 'PUNGGOL': 17, 'WOODLANDS': 24, 'YISHUN': 25,
             'TAMPINES': 22, 'JURONG WEST': 13, 'BEDOK': 1, 'HOUGANG': 11,
             'CHOA CHU KANG': 8, 'ANG MO KIO': 0, 'BUKIT MERAH': 4, 'BUKIT PANJANG': 5,
             'BUKIT BATOK': 3, 'TOA PAYOH': 23, 'PASIR RIS': 16, 'KALLANG/WHAMPOA': 14,
             'QUEENSTOWN': 18, 'SEMBAWANG': 19, 'GEYLANG': 10, 'CLEMENTI': 9,
             'JURONG EAST': 12, 'BISHAN': 2, 'SERANGOON': 21, 'CENTRAL AREA': 7,
             'MARINE PARADE': 15, 'BUKIT TIMAH': 6},
    'flat_type': {'4 ROOM': 3, '5 ROOM': 4, '3 ROOM': 2,
                  'EXECUTIVE': 5, '2 ROOM': 1, 'MULTI-GENERATION': 6,
                  '1 ROOM': 0},
    'storey_range': {'04 TO 06': 1, '07 TO 09': 2, '10 TO 12': 3, '01 TO 03': 0,
                     '13 TO 15': 4, '16 TO 18': 5, '19 TO 21': 6, '22 TO 24': 7,
                     '25 TO 27': 8, '28 TO 30': 9, '31 TO 33': 10, '34 TO 36': 11,
                     '37 TO 39': 12, '40 TO 42': 13, '43 TO 45': 14, '46 TO 48': 15,
                     '49 TO 51': 16},
    'flat_model': {'Model A': 8, 'Improved': 5, 'New Generation': 12, 'Premium Apartment': 13,
                   'Simplified': 16, 'Apartment': 3, 'Maisonette': 7, 'Standard': 17,
                   'DBSS': 4, 'Model A2': 10, 'Model A-Maisonette': 9, 'Adjoined flat': 2,
                   'Type S1': 19, 'Type S2': 20, 'Premium Apartment Loft': 14, 'Terrace': 18,
                   'Multi Generation': 11, '2-room': 0, 'Improved-Maisonette': 6, '3Gen': 1,
                   'Premium Maisonette': 15},
}

# Function to display the home page
def About_page():

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.subheader(":violet[Problem Statement:]")
        st.write("""
        - Develop a machine learning model and build a user-friendly web application to predict resale flat prices in Singapore, 
          assisting both potential buyers and sellers in estimating the market value based on historical transaction data.
        """)

    with col2:

        st.write("* **:red[Purpose]** : Predict the selling Flat Price.")
        st.write("* **:red[Techniques Used]** : Data Wrangling and Preprocessing, Exploratory Data Analysis (EDA), Model Building and Evaluation, Web Application Development.")
        st.write("* **:red[Algorithm]** : Random Forest Regression.")

    st.image(Image.open(r"D://6_Singapore Resale Flat Prices Predicting//Images//HDB_FLAT.png"), width=1000)    

# Function to display the flat prediction page
def Prediction_page():

    input_data = {}
    current_year = datetime.now().year

    col1, col2, col3 = st.columns(3)

    with col1:

        input_data['town'] = st.selectbox('Town:', options=list(categorical_mappings['town'].keys()))
        input_data['flat_type'] = st.selectbox('Flat Type:', options=list(categorical_mappings['flat_type'].keys()))
        input_data['storey_range'] = st.selectbox('Storey Range:', options=list(categorical_mappings['storey_range'].keys()))
        input_data['floor_area_sqm'] = st.number_input('Floor Area (sqm):', min_value=0.0, max_value=173.0, value=60.0, step=1.0)
        input_data['flat_model'] = st.selectbox('Flat Model:', options=list(categorical_mappings['flat_model'].keys()))

    with col2:

        input_data['year'] = st.selectbox('Year:', options=list(range(2015, current_year + 1)))
        input_data['month_of_year'] = st.selectbox('Month of Year:', options=list(range(1, 13)))
        input_data['lease_commence_year'] = st.selectbox('Lease Commence Year:', options=list(range(1900, 2101)))
        input_data['remaining_lease_years'] = st.slider('Remaining Lease Years:', min_value=0, max_value=99, step=1)
        input_data['remaining_lease_months'] = st.slider('Remaining Lease Months:', min_value=0, max_value=11, step=1)

    with col3:

        if st.button("Predict"):

            # Convert categorical variables to numerical using mappings
            input_array = np.array([
                categorical_mappings['town'][input_data['town']],
                categorical_mappings['flat_type'][input_data['flat_type']],
                categorical_mappings['storey_range'][input_data['storey_range']],
                input_data['floor_area_sqm'],
                categorical_mappings['flat_model'][input_data['flat_model']],
                input_data['year'],
                input_data['month_of_year'],
                input_data['lease_commence_year'],
                input_data['remaining_lease_years'],
                input_data['remaining_lease_months']
            ]).reshape(1, -1)

            # Perform prediction
            prediction = model.predict(input_array)

            # Display the prediction result
            prediction_scale = np.exp(prediction[0])

            st.subheader("Prediction Result:")

            # st.write(f"The Predicted Resale Flat price = {prediction_scale:,.2f} INR")

            st.markdown(f"""
            <div style='border: 2px solid #009999; border-radius: 10px; padding: 10px;'>
                <p style='font-size: 1.2em; color: white; background-color: #009999; padding: 5px; border-radius: 5px;'>
                    The Predicted Resale Flat price = <span style='font-size: 1.5em; color: yellow; background-color: black; padding: 5px; border-radius: 5px;'>{prediction_scale:,.2f} INR</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

# Function to display the analysis page
def Conclusion_page():

    col1, col2 = st.columns(2, gap="large")
    
    with col1:

        st.subheader(":yellow[Model Performance]")

        st.write("""
        The accuracy of various machine learning models was evaluated:
        - Linear Regressor Accuracy: **:red[68.23%]**
        - Random Forest Regressor Accuracy: **:red[95.66%]**
        - Decision Tree Regressor Accuracy: **:red[92.10%]**

        Based on the accuracy scores, the **:red[Random Forest Regressor]** was chosen as the final model due to its highest accuracy of **:red[95.66%]**.
        """)

    with col2:

        st.subheader(":yellow[Final Observations]")

        st.write("""
        -  Empowered with accurate price estimates, potential buyers can make informed decisions, enhancing their purchasing strategies.
        -  Sellers: Sellers benefit from pricing guidance, aiding in competitive pricing and effective negotiation.
        -  Real Estate Professionals: The project serves as a valuable tool for real estate agents and analysts, facilitating market analysis and decision-making processes.
        """)

    st.image(Image.open(r"D://6_Singapore Resale Flat Prices Predicting//Images//Singapore_Flat.jpg"), width=700) 

# Configure the Streamlit app layout and style
streamlit_config()

style_submit_button()

style_prediction()

# Create the navigation menu in the sidebar


# Create the navigation menu in the sidebar
with st.sidebar:

    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Flat Prediction", "Analysis"],
        icons=["house", "graph-up", "bar-chart"],
        menu_icon="cast",
        default_index=0,
    )

# Display the selected page
if selected == "Home":
    About_page()

elif selected == "Flat Prediction":
    Prediction_page()

elif selected == "Analysis":
    Conclusion_page()

# ---------------------------------------------         END           ---------------------------------------------------