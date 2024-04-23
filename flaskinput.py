import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string, redirect, url_for
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
    for index, row in user_df.iterrows():
        matches = main_df[(main_df['Column1'] == row['Column1']) & (main_df['Column2'] == row['Column2'])]
        if not matches.empty:
            return f"Class label = '{matches.iloc[0]['Class']}'"
    return "Class label = 'No match found'"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            excel_path = filepath.replace('.npy', '.xlsx')
            convert_npy_to_excel(filepath, excel_path)
            main_df = load_excel_data('combined_data_classified.xlsx')  # Adjust the path accordingly
            user_df = load_excel_data(excel_path)
            result = find_first_matching_class_label(main_df, user_df)
            return render_template_string('<h1>{{ result }}</h1>', result=result)
        else:
            return 'Invalid file format or no file selected'
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload a Numpy file</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
