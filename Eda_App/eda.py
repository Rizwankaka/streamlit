import numpy as np 
import pandas as pd
import seaborn as sns 
import streamlit as st 
from ydata_profiling import ProfileReport
import streamlit_pandas_profiling as sp
from streamlit_pandas_profiling import st_profile_report

# webapp ka title 
st.markdown('''
# **Exploratory Data Analysis web application**
This app is developed by Rizwan Rizwan called **EDA App**
            ''')

# how to upload a file from pc 

with st.sidebar.header('Upload your dataset(.csv)'):
    uploaded_file=st.sidebar.file_uploader('Upload your file', type=['csv'])
    df=sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv)")

# profiling report for pandas 
if uploaded_file is not None:
    @st.cache_data
    def load_csv():
        csv=pd.read_csv(uploaded_file)
        return csv
    df=load_csv()
    pr=ProfileReport(df, explorative=True)
    st.header('**Input DF**')
    st.write(df)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)
else:
    st.info('Awaiting for CSV file, upload ker bhe do ya kaam nhe lena?')
    if st.button('Press to use example data'):
        # example dataset 
        @st.cache_data
        def load_data():
            a= pd.DataFrame(np.random.rand(100,5), 
                            columns=['age', 'banana', 'codanics', 'deutschland', 'ear'])

            return a
        df=load_data()
        pr=ProfileReport(df, explorative=True)
        st.header('**Input DF**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)