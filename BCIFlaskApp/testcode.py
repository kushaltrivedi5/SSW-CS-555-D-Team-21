from flask import Flask, request, render_template_string, redirect, url_for, send_from_directory
import os
import papermill as pm

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            process_notebook(filepath)
            return redirect(url_for('download_file', filename='output.html'))
        else:
            return 'Invalid file format or no file selected'
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def process_notebook(input_filepath):
    nb_path = 'path/to/2. BCI EEG Classification.ipynb'
    output_nb_path = os.path.join(UPLOAD_FOLDER, 'output.html')
    pm.execute_notebook(
        nb_path,
        output_nb_path,
        parameters={'input_file_path': input_filepath}
    )

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
