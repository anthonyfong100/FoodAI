#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
from scipy.misc import imsave, imread, imresize
from model.load import * 
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input 
from werkzeug.utils import secure_filename
import numpy as np
import keras.models, re, sys, os, cv2
from flask import jsonify



sys.path.append(os.path.abspath("./model"))
app = Flask(__name__)
global model, graph
model, graph, class_dictionary= init()



#decoding an image from bytes into numpy array and do the required preprocessing required for inceptionv3
def _preprocess_img(image):
	image_decoded = cv2.imdecode(np.frombuffer(image,np.uint8),-1)
	image = cv2.resize(image_decoded, (299, 299)) 
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)   #we need to change from BGR to RGB
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)
	return image


@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")


@app.route('/api/predict/',methods=['POST'])
def predict():
	f = request.files['file']
	image = f.read()
	image = _preprocess_img(image)

	topNum=5
	with graph.as_default():
		y_pred_prob = model.predict(image)
		cls = np.argmax(y_pred_prob, axis=1)
		top_n_preds_ix = np.argpartition(y_pred_prob, -topNum)[:,-topNum:]

		#convert from ix to class label
		#reverse the list, try to use numpy inbuilt to list function because cant jsonify a list with numpy float32 variable
		inv_map = {v: k for k, v in class_dictionary.items()}
		label = np.array([inv_map[x] for x in top_n_preds_ix[0]][::-1]).tolist() 
		probability = np.array([y_pred_prob[0,x] for x in top_n_preds_ix[0]][::-1]).tolist()
		result = dict(zip(label,probability))
	return jsonify(result)





if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)