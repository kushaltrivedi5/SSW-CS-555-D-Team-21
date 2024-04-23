import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string, redirect, url_for
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'npy'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_npy_to_excel(npy_path, excel_path):
    """Load .npy file and save as .xlsx after validating the expected structure."""
    data_array = np.load(npy_path, allow_pickle=True)
    if data_array.shape[1] == 3:  # Ensure the array has exactly three columns
        df = pd.DataFrame(data_array, columns=['Column1', 'Column2', 'Column3'])
    else:
        raise ValueError(f"Unexpected number of columns in data: {data_array.shape[1]}")
    df.to_excel(excel_path, index=False)
    print(f"Converted {npy_path} to Excel and saved as {excel_path}")

def load_excel_data(file_path):
    """Load an Excel file into a DataFrame."""
    return pd.read_excel(file_path)

def find_matching_class_label(main_df, user_df):
    """Compare user_df with main_df and return matching class labels from Column3."""
    results = []
    for _, row in user_df.iterrows():
        match = main_df[(main_df['Column1'] == row['Column1']) & (main_df['Column2'] == row['Column2'])]
        if not match.empty:
            results.append(match.iloc[0]['Column3'])  # Extracting the label from Column3
        else:
            results.append("No match found")
    return '<br>'.join(results)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            excel_path = filepath.replace('.npy', '.xlsx')
            try:
                convert_npy_to_excel(filepath, excel_path)
                user_df = load_excel_data(excel_path)
                main_df = load_excel_data('combined_data_classified.xlsx')
                result = find_matching_class_label(main_df, user_df)
                return render_template_string('<h1>Results:<br>{{ result }}</h1>', result=result)
            except Exception as e:
                return render_template_string(f'<h1>Error: {{error}}</h1>', error=str(e))
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
