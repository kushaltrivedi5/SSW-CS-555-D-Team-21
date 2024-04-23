import pandas as pd
import numpy as np

# Load the Excel file
df = pd.read_excel('help_me_data.xlsx')

# Convert the DataFrame to a NumPy array
data_array = df.values

# Save the array as a .npy file
np.save('helpme_filename.npy', data_array)

# Print confirmation message
print("The .npy file has been successfully created.")
