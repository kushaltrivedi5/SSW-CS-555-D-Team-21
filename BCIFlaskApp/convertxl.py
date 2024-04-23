import os

def rename_files_in_directory(directory_path):
    """
    This function renames all .xlsx files in the specified directory to .npy files.
    """
    # List all files in the given directory
    for filename in os.listdir(directory_path):
        # Check if the current file is an Excel file
        if filename.endswith('.xlsx'):
            # Create the new filename by replacing the old extension with .npy
            new_filename = filename.replace('.xlsx', '.npy')
            # Construct full file paths
            original_filepath = os.path.join(directory_path, filename)
            new_filepath = os.path.join(directory_path, new_filename)
            # Rename the file
            os.rename(original_filepath, new_filepath)
            print(f"Renamed {filename} to {new_filename}")

# Example usage: replace 'path/to/directory' with the path to your directory containing the Excel files
directory_path = 'path/to/directory'
rename_files_in_directory(directory_path)
