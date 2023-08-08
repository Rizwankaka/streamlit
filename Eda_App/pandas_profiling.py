import pandas as pd 
import ydata_profiling
import streamlit as st 
import seaborn as sns 

from streamlit_pandas_profiling import st_profile_report

df=sns.load_dataset('titanic')
pr=df.profile_report()
st_profile_report(pr)