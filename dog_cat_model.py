import cv2 
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model

from flask import Flask,render_template,redirect,request
app=Flask(__name__)




CATEGORIES=['Dog','Cat']


def prepare(image):
    img_size=100
    img_array=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(img_size,img_size))
    return new_array.reshape(-1,img_size,img_size,1)

model = tf.keras.models.load_model(r"K:\kagglecatsanddogs_3367a\Dogs_vs_Cats_500.model")







@app.route('/')
def hello():
    return render_template("dog_cat.html")



@app.route("/home")
def home():
    return redirect('/')

@app.route('/submit',methods=['POST'])
def submit_data():
    if request.method == 'POST':
        
        f=request.files['userfile']
        f.save(f.filename)
        print(f)
        image=f.filename

        prediction=model.predict([prepare(image)/255.0])
        
        print(round(prediction[0][0]))
        print(CATEGORIES[int(round(prediction[0][0]))])
        
        img=mpimg.imread(image)
        imgplot=plt.imshow(img)
        plt.title(CATEGORIES[int(prediction[0][0])])
        plt.show()
        msg=CATEGORIES[int(round(prediction[0][0]))]
        
        
        
        return  render_template("dog_cat.html" , msg=msg)
    

if __name__ =="__main__":
    
    
    app.run()