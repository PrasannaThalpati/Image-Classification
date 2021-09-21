from flask import Flask, render_template, request

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

import numpy as np
import os

app = Flask(__name__)
model = load_model('cnn_img_model.h5')

def model_predict(img_path,model):

    test_image=image.load_img(img_path,target_size=(64,64))
    test_image=image.img_to_array(test_image)
    test_image=test_image/255
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/success', methods=['GET','POST'])
def upload_file():
    if request.method =='POST':
        f = request.files['my_image']
        basepath = os.path.dirname(os.path.realpath('__file__'))
        file_path = os.path.join(basepath,'Static/upload_img', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        result = model_predict(file_path,model)
        print(result)
        if (result > 0.5):
            return render_template('index.html', pred="It's a Dog...")
        else:
            return render_template('index.html', pred1="It's a Cat...")



if __name__ =='__main__':
    app.run(debug=True)

