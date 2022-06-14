from flask import Flask, render_template, request
from PIL import Image
import pickle
import numpy as np


app = Flask(__name__, static_url_path='/static')


@app.route('/', methods=['GET']) # 루트를 GET으로 호출하면, 
def index():
    return render_template('index.html')

@app.route('/mnist/', methods=['GET', 'POST'])
def mnist():
    if request.method == 'GET':
        return render_template('mnist_form.html')
    else:
        if request.files['mnist_file'].filename == '':
            return render_template('mnist_form.html')

        f = request.files['mnist_file']
        path = './static/data/' + f.filename
        name = f.filename
        f.save(path)
        
        img = Image.open(path).convert('L')
        img = np.resize(img, (1, 784))
        img = 255 - img

        f = open('./model.pickle', 'rb')
        model = pickle.load(f)
        f.close()

        pred = model.predict(img)
        print(path)
        return render_template('mnist_result.html', img=name, pred=pred)




# @app.route('/test/')
# def test():
#     return 'test'

# @app.route('/test2/')
# def test2():
#     return 'test2'

# @app.route('/test2/test22/')
# def test22():
#     return 'test2/22'




if __name__ == '__main__':
    app.run(debug=True)