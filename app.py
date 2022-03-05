# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:31:13 2022

@author: Lenovo
"""

from flask import Flask
app = Flask(__name__)


from flask import request, render_template
import joblib
import pickle

from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method=="POST":
        Nikkei = request.form.get("Nikkei")
        print(Nikkei)
        
        model1 = joblib.load("STI_REG")
        pred1 = model1.predict([[Nikkei]])
        str1 = "The prediction for STI using Regression is: "+ str(pred1)
        
        model2 = joblib.load("STI_DT")
        pred2 = model2.predict([[Nikkei]])
        str2 = "The prediction for STI using Decision Tree is: "+ str(pred2)
        
        make_keras_picklable()
        with open('STI_NN', 'rb') as f:
            model3 = pickle.load(f)
        #model3 = joblib.load("STI_NN")
        pred3 = model3.predict([[Nikkei]])
        str3 = "The prediction for STI using Neural Network is: "+ str(pred3)

        model4 = joblib.load("STI_SVM")
        pred4 = model4.predict([[Nikkei]])
        str4 = "The prediction for STI using SVM is: "+ str(pred4)
        
        return(render_template("index.html",result1 = str1,result2=str2,result3=str3,result4=str4))
    
    else:
        return(render_template("index.html",result1 = "2",result2 = "2",result3 = "2",result4 = "2"))
    

if __name__ == "__main__":
    app.run()
