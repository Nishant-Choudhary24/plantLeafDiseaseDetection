from django.shortcuts import render

# Create your views here.
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from tensorflow import Graph
import matplotlib.pyplot as plt

import json

img_height, img_width=128,128
with open('./models/lab.json','r') as f:
    labelInfo=f.read()

labelInfo=json.loads(labelInfo)

model=load_model('./models/inceptionv3_50.h5')

def index (request):
    context ={'a':1}
    return render(request,'index.html',context)

def predictImage(request):
    # plt.imshow(request)
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)

    testimage='.'+filePathName
    import numpy as np
    import matplotlib.image as mpimg


    img = mpimg.imread(testimage)


    plt.imshow(img)

    img=tf.image.resize(img , [128 , 128])

    img=np.resize(img , [1 ,128 , 128 , 3])
    img = img/255
    pre=model.predict(img)

    predictedLabel =labelInfo[str(np.argmax(pre))]
    # img = image.load_img(testimage, target_size=(img_height, img_width))

    # x = image.img_to_array(img)

    # x=x.reshape(1,img_height, img_width,3)
    # predi=model.predict(x)

    
    # predictedLabel=labelInfo[str(np.argmax(predi[0]))]

    context={'filePathName':filePathName,'predictedLabel':predictedLabel}
    return render(request,'index.html',context)