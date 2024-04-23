import pandas as pd
import numpy as np

def convert_excel_to_npy(excel_path, npy_path):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(excel_path)
    
    # Convert the DataFrame to a NumPy array
    np_array = df.to_numpy()
    
    # Save the NumPy array to a .npy file
    np.save(npy_path, np_array)
    print(f"Data saved to {npy_path} successfully.")

# Specify the path to the Excel file and the desired .npy file path
excel_path = 'user4.xlsx'
npy_path = 'user4.npy'

# Perform the conversion
convert_excel_to_npy(excel_path, npy_path)
