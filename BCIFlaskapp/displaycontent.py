import numpy as np
import pandas as pd

def convert_npy_to_excel(npy_path, excel_path):
    # Load the data from .npy file
    data = np.load(npy_path, allow_pickle=True)
    
    # Convert the numpy array to a pandas DataFrame
    # Ensure you adjust the column names according to the structure of your numpy array
    if data.ndim == 1:
        df = pd.DataFrame(data, columns=['Column1'])
    elif data.shape[1] == 2:
        df = pd.DataFrame(data, columns=['Column1', 'Column2'])
    elif data.shape[1] == 3:
        df = pd.DataFrame(data, columns=['Column1', 'Column2', 'Column3'])
    else:
        df = pd.DataFrame(data)
    
    # Save the DataFrame to an Excel file
    df.to_excel(excel_path, index=False)
    print(f"Data from {npy_path} has been written to {excel_path} successfully.")

# Example usage
npy_file_path = 'hello_data.npy'
excel_file_path = 'hello_output.xlsx'
convert_npy_to_excel(npy_file_path, excel_file_path)
