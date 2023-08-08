import streamlit as st

st.header('Original code')

first_val = st.number_input('Enter number:')
second_val = st.number_input('Enter second number')
add_val = first_val + second_val
st.write(add_val)


st.header('Revised code')

with st.form('my_form'):
  first_value = st.number_input('Enter number:')
  second_value = st.number_input('Enter second number:')

  submitted = st.form_submit_button('Submit')
  if submitted:
      add_value = first_value + second_value
      #st.write(add_value)

if submitted:
  st.write(add_value)