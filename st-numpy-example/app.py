import streamlit as st
import numpy as np

st.title('🔢 NumPy - A minimum working example')

# Create a 3 by 3 data matrix using NumPy
data_matrix = np.array([[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]])

# Display the contents of the created data matrix variable
st.write('The following is a data matrix created using NumPy')
st.write(data_matrix)