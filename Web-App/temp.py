import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
from fastai import *
from fastai.vision import *

import matplotlib as plt

app = Flask(__name__)
classes = [0,1,2,3,4]
#path = os.curdir()
#databunch = ImageList(train,"E:\my programs\python\App\data").split_none().transform(tfms).databunch()
data_bunch = ImageDataBunch.single_from_classes("E:\my programs\python\App",classes,ds_tfms=get_transforms(), size=150).normalize(imagenet_stats)
learn = cnn_learner(data_bunch, models.resnet101, pretrained=False)
learn.load('dr')

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        #return redirect(url_for('prediction', filename=filename))
        return prediction(filename)
    return render_template('index.html')

@app.route('/prediction/<filename>')
def prediction(filename):
    
    my_image = open_image(os.path.join('uploads', filename))
    #Step 2
    Tf = partial(Image.apply_tfms,tfms=get_transforms(do_flip=True, flip_vert = True)[0][1:]+get_transforms(do_flip=True, flip_vert = True)[1],size = 512)
    #img = open_image(my_image)
    pre = learn.predict(my_image)
    x = pre[1]
    x = int(x)
    #number_to_class = tensor([0,1,2,3])
    
    cat = ['NO-DR','Mild','Moderate','Severe','Proleferative DR']
    #index = np.argsort(x)
    
    predictions={"class0":str(x),
                 
                 "prob0":[str(x),cat[x]]
    }
    return render_template('predict.html', predictions=predictions)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 9000, app,use_debugger=True,use_reloader=False)
#app.run(host='0.0.0.0', port=80,)