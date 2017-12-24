#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 00:33:34 2017

@author: purna
"""
import os
import os.path
import cv2
import glob
import imutils
import numpy as np
from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit


def extract_single_letter():
    captcha_image_files = glob.glob(os.path.join("generated_captcha_images","*"))
    counts = {}
    print(type(captcha_image_files))
    for (i,captcha_image_file) in enumerate(captcha_image_files):
        print("[INFO] processing image {}/{}".format(i+1,len(captcha_image_files)))
        filename = os.path.basename(captcha_image_file)
        #print(filename)
        captcha_correct_text = os.path.splitext(filename)[0]
        image = cv2.imread(captcha_image_file)
        #print(np.asarray(image).shape)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray,20,20,20,20,cv2.BORDER_REPLICATE)
        thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)[1]
        conters = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        conters = conters[0] if imutils.is_cv2() else conters[1]
        
        letter_image_regions = []
        for conter in conters:
            (x,y,w,h) = cv2.boundingRect(conter)
            
            if w / h > 1.25:
                half_width = int(w/2)
                letter_image_regions.append((x,y,w,h))
                letter_image_regions.append((x+half_width,y,half_width,h))
            else:
                letter_image_regions.append((x,y,w,h))
            
            if len(letter_image_regions) != 4:
                continue
            letter_image_regions = sorted(letter_image_regions,key = lambda x: x[0])
            
            for letter_bounding_box,letter_text in zip(letter_image_regions,captcha_correct_text):
                x,y,w,h = letter_bounding_box
                print(letter_text)
                letter_image = gray[y-2:y+h+2,x-2:x+w+2]
                save_path = os.path.join("extracted_letter_images",letter_text)
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                count = counts.get(letter_text,1)
                p = os.path.join(save_path,"{}.png".format(str(count).zfill(6)))
                cv2.imwrite(p, letter_image)
                
                counts[letter_text] = count + 1

def tensor():
    try:
        tensor = np.load("tensor.npy")
        labels = np.load("labels.npy")
    except:
        data = []
        labels = []
        
        for image_file in paths.list_images("extracted_letter_images"):
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = resize_to_fit(image, 20, 20)
            image = np.expand_dims(image, axis=2)
            label = image_file.split(os.path.sep)[-2]
            data.append(image)
            labels.append(label)
        tensor = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)
    return tensor,labels

def prediction(lb,model_made):
    
    captcha_image_files = list(paths.list_images("generated_captcha_images"))
    captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)
    for image_file in captcha_image_files:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]
    
        letter_image_regions = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w / h > 1.25:
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                letter_image_regions.append((x, y, w, h))
        if len(letter_image_regions) != 4:
            continue
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
        output = cv2.merge([image] * 3)
        predictions = []
        for letter_bounding_box in letter_image_regions:
            x, y, w, h = letter_bounding_box
            letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
            letter_image = resize_to_fit(letter_image, 20, 20)
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)
            prediction = model_made.predict(letter_image)
            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)
            cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
            cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        captcha_text = "".join(predictions)
        print("CAPTCHA text is: {}".format(captcha_text))
        plt.imshow(output, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
          
def model():
    model = Sequential()
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dense(32, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

def model_call(X_train,X_test,Y_train,Y_test):
    try:
        model = load_model("captcha_model_try1.hdf5")
    except:
        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)
        model.save("captcha_model_try1.hdf5")
    return model
    

if __name__ == "__main__":
   print("a")
   #extract_single_letter()
   tensor,labels = tensor()
   (X_train, X_test, Y_train, Y_test) = train_test_split(tensor, labels, test_size=0.25, random_state=0)
   lb = LabelBinarizer().fit(Y_train)
   Y_train = lb.transform(Y_train)
   Y_test = lb.transform(Y_test)
   print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
   model =  model()
   #print(model.summary)
   model_made = model_call(X_train,X_test,Y_train,Y_test)
   prediction(lb,model_made)
    
