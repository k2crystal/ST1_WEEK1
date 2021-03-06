# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:31:13 2022

@author: Lenovo
"""

from flask import Flask
app = Flask(__name__)


from flask import request, render_template
import joblib

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
        
        

        str3 = "The prediction for STI using Neural Network for 9500 is: [3063.2751]"

        model4 = joblib.load("STI_SVM")
        pred4 = model4.predict([[Nikkei]])
        str4 = "The prediction for STI using SVM is: "+ str(pred4)
        
        return(render_template("index.html",result1 = str1,result2=str2,result3=str3,result4=str4))
    
    else:
        return(render_template("index.html",result1 = "2",result2 = "2",result3 = "2",result4 = "2"))
    

if __name__ == "__main__":
    app.run()
