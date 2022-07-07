# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:31:14 2022

@author: KarthickAnu
"""

import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

ineuron_path =open('ineuronmodel.pkl','rb')
ineuron_project =pickle.load(ineuron_path)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html = True)

def main():
    st.title('Insurance Premium Prediction App')
    st.markdown('Just Enter the following details and we will predict the premium amount for the insured person')
    a = st.slider("Age(years)",1,100)
    b = st.selectbox("Gender",('female','male'))
    if b == 'female':
        b=0
    else:
        b=1
    c = st.number_input("BMI",min_value=16.0, max_value=54.0,step=0.1)
    d = st.number_input('Children', min_value=0, max_value=5)
    e = st.selectbox("smoker",('yes','no'))
    if e == 'no':
       e=0
    else:
       e=1
    f = st.selectbox("Region",('northeast','northwest','southeast','southwest'))
    if f == 'northeast':
        f=0
    elif f == 'northwest':
        f=1
    elif f == 'southeast':
        f=2
    else:
        f=3
    
    
    submit = st.button('Predict Expenses')
    if submit: 
       prediction = ineuron_project.predict([[a,b,c,d,e,f]])
       st.write('Insurance expenses are', prediction)
if __name__ == '__main__':
    main()
