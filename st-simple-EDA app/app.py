# Simple EDA app
# By Rizwan Rizwan (Data Scientist)
# https://youtube.com/Rizwan Rizwan

# Importing requisite libraries
import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
     page_title='Simple EDA App',
     page_icon='🎈',
     layout='wide',
     initial_sidebar_state='expanded')

# Title of the app
st.title('🎈 Simple EDA App')

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/iris.csv')

# EDA
st.subheader('EDA')

data_describe = df.describe()
st.write(data_describe)

data_sepal_length = pd.qcut(df['Sepal.Length'], 3, labels=["low", "medium", "high"]).value_counts()
data_sepal_width = pd.qcut(df['Sepal.Width'], 3, labels=["low", "medium", "high"]).value_counts()
data_petal_length = pd.qcut(df['Petal.Length'], 3, labels=["low", "medium", "high"]).value_counts()
data_petal_width = pd.qcut(df['Petal.Width'], 3, labels=["low", "medium", "high"]).value_counts()
data_species = df.Species.value_counts()

eda_col1, eda_col2, eda_col3, eda_col4, eda_col5 =  st.columns(5)
with eda_col1:
  st.bar_chart(data_sepal_length)
with eda_col2:
  st.bar_chart(data_sepal_width)
with eda_col3:
  st.bar_chart(data_petal_length)
with eda_col4:
  st.bar_chart(data_petal_width)
with eda_col5:
  st.bar_chart(data_species)