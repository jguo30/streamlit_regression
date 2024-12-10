import streamlit as st
import pandas as pd
import pickle

st.header('House Price Predicter')
st.write('Please enter information about the house you want to buy.')

gr_area = st.number_input('Please enter the above ground area of the house.')
lot_area = st.slider('Please select the total lot area', min_value = 0, max_value = 1000)
over_qual = st.selectbox('Please select the overall quality of the house',{'Above_Average', 'Average', 'Good', 'Very_Good', 'Excellent', 'Below_Average', 'Fair', 'Poor', 'Very_Excellent', 'Very_Poor'})
sale_con = st.selectbox('Please select the sale condition of the house', {'Normal', 'Partial', 'Family', 'Abnorml', 'Alloca', 'AdjLand'})

X = pd.DataFrame({'Gr_Liv_Area': gr_area,
                 'Overall_Qual': over_qual, 
                 'Sale_Condition': sale_con, 
                 'Lot_Area': lot_area}, index=[0])

st.dataframe(X)

with open('lr_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
preds = model.predict(X)

house_price = st.write('Your house is valued at:', preds)
