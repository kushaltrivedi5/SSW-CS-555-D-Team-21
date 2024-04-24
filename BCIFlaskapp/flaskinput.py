import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'npy'}  # Allow only .npy files

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_npy_to_excel(npy_path, excel_path):
    data_array = np.load(npy_path, allow_pickle=True)
    # Create DataFrame from Numpy Array
    df = pd.DataFrame(data_array, columns=['Column1', 'Column2', 'Column3'] if data_array.shape[1] == 3 else None)
    df.to_excel(excel_path, index=False)

def load_excel_data(file_path):
    """Load an Excel file into a DataFrame."""
    return pd.read_excel(file_path)

def find_first_matching_class_label(main_df, user_df):
    """Find the first matching class label in main_df for entries in user_df."""
    result = []
    for index, row in user_df.iterrows():
        matches = main_df[(main_df['Column1'] == row['Column1']) & (main_df['Column2'] == row['Column2'])]
        if not matches.empty:
            result.append(matches.iloc[0]['Class'])  # Append only the value of "Class" column
        else:
            result.append('No match found')
    return result

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        excel_path = filepath.replace('.npy', '.xlsx')
        convert_npy_to_excel(filepath, excel_path)
        main_df = load_excel_data('combined_data_classified.xlsx')  # Adjust the path accordingly
        user_df = load_excel_data(excel_path)
        result = find_first_matching_class_label(main_df, user_df)
        return jsonify({'result': result})
    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
