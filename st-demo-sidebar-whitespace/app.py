# import modules
import streamlit as st

st.markdown("""
  <style>
    .css-o18uir.e16nr0p33 {
      margin-top: -75px;
    }
  </style>
""", unsafe_allow_html=True)

# configure sidebar text and widgets
st.sidebar.title("My App")
file = st.sidebar.file_uploader("Choose a file", key='a')
cb1 = st.sidebar.checkbox('1', key='b')
cb2 = st.sidebar.checkbox('2', key='c')
cb3 = st.sidebar.checkbox('3', key='d')
cb4 = st.sidebar.checkbox('4', key='e')
cb5 = st.sidebar.checkbox('5', key='f')
cb7 = st.sidebar.checkbox('6', key='g')
cb8 = st.sidebar.checkbox('7', key='h')
cb9 = st.sidebar.checkbox('8', key='i')
cb10 = st.sidebar.checkbox('9', key='j')
start_btn = st.sidebar.button('Run', key='k')
st.sidebar.text("")