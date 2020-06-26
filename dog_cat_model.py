import cv2 
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model

from flask import Flask,render_template,redirect,request

import os


app=Flask(__name__)

app.config["IMAGE_UPLOADS"] = "static/img/"


CATEGORIES=['Dog','Cat']


def prepare(image):
    img_size=100
    img_array=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(img_size,img_size))
    return new_array.reshape(-1,img_size,img_size,1)

model = tf.keras.models.load_model(r"Dogs_vs_Cats_500_final.model")







@app.route('/')
def hello():
    return render_template("dog_cat.html")



@app.route("/home")
def home():
    return redirect('/')

@app.route('/submit_cdc',methods=['POST'])
def submit_data():
    if request.method == 'POST':
        
        f=request.files['userfile']
        f.save(os.path.join(app.config["IMAGE_UPLOADS"], f.filename))

        image=os.path.join(app.config["IMAGE_UPLOADS"], f.filename)

        prediction=model.predict([prepare(image)/255.0])



        
        img=mpimg.imread(image)
        imgplot=plt.imshow(img)
        plt.title(CATEGORIES[int(prediction[0][0])])
        plt.show()
        msg=CATEGORIES[int(round(prediction[0][0]))]
        full_filename= os.path.join(app.config["IMAGE_UPLOADS"], f.filename)
        
        
        return  render_template("dog_cat_img.html" , msg=msg,user_image = full_filename)
    

if __name__ =="__main__":
    
    
    app.run()