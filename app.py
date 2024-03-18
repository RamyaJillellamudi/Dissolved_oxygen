# from flask import Flask, render_template, request
# import numpy as np
# import tensorflow as tf
# import pickle

# app = Flask(__name__)

# # Load the trained model
# model= pickle.load(open("model.pkl", "rb"))
# # try:
# #     loaded_model = tf.keras.models.load_model(model)
# # except OSError:
# #     print("Error loading Keras model. Model file does not exist")
# #     loaded_model = None
# # except ValueError as error:
# #     print(f"Error loading Keras model: {error}")   
# #     loaded_model = None

# # Function to preprocess input data for prediction
# def preprocess_input(data):
#     processed_data = np.array(data).reshape(1, 1, len(data))
#     return processed_data
# @app.route('/')
# def home():
# 	return render_template('index.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def index():
#     result = None

#     if request.method == 'POST' and model is not None:
#         # Get user input from the form
#         user_input = [
#             float(request.form['do']),
#             float(request.form['temp']),
#             float(request.form['ph']),
#             float(request.form['turbidity']),
#             float(request.form['conductivity']),
#             float(request.form['bod']),
#             float(request.form['nitrate']),
#             float(request.form['coliform'])
#         ]

#         # Preprocess the input data
#         processed_input = preprocess_input(user_input)
#         print(processed_input)

#         try:
#             # Make predictions using the loaded model
#             predictions = model.predict(processed_input)
#             print("predictions:",predictions)
#             # predicted_label = np.argmax(predictions)
#             result = "Contaminated" if predictions[0,0] >= 0.5 else "Non-Contaminated"
#             print(result)

#             # do_level= predictions[0, 0]
#             # threshold = 0.5
#             # if do_level >= threshold:
#             #     result= "Alert: Dissolved Oxygen Level Exceeds Safe Limit!"
#             # else:
#             #     result = "Dissolved Oxygen Level within Safe Limit."
#         except Exception as e:
#             print(f"Error making predictions: {e}")

#             # Get the predicted label (0 or 1)
            

#             # Map the label to "contaminated" or "non-contaminated"
            
#         return render_template('result.html', result=result)

# if __name__ == '__main__':
#     app.run(debug=True)
# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import joblib

# # Create flask app
# flask_app = Flask(__name__)
# model = joblib.load(open("model.pkl", "rb"))

# @flask_app.route("/")
# def Home():
#     return render_template("index.html")

# @flask_app.route("/predict", methods = ["POST"])
# def predict(): 
#     result=None
#     if request.method == 'POST':
        
#          # Get user input from the form
#         user_input= [
#              request.form['do'],
#              request.form['temp'],
#              request.form['ph'],
#              request.form['turbidity'],
#              request.form['conductivity'],
#              request.form['bod'],
#              request.form['nitrate'],
#              request.form['coliform']
#          ]

#         float_features = [float(x) for x in request.form.values()]
#         features = [np.array(float_features)]
#         prediction = model.predict(features)
#         if prediction==0:
#             result="No Alert"
#         elif prediction==1:
#             result='Alert the D.O is more'
#     return render_template("result.html", result=result)
# # # @flask_app.route('/results',methods=['POST'])
# # # def results():
# # #     data = request.get_json(force=True)
# # #     prediction= model.predict([np.array(list(data.values()))])
# # #     if prediction==0:
# # #         output="No Alert"
# # #     elif prediction==1:
# # #         output='Alert the D.O is more'
# # #     return jsonify(output)
    
# #     # output='{0:.{1}f}'.format(prediction[0][1], 2)

# #     # if output>str(7.0):
# #     #     return render_template('index.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
# #     # else:
# #     #     return render_template('index.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")


# if __name__ == "__main__":
#     flask_app.run(debug=True)
# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import pickle
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load the scaler and model
# model = pickle.load(open('model.pkl', 'rb'))

# # Define a route for the home page
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Define a route for prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the input data from the form
#     features = [float(x) for x in request.form.values()]

#     # Transform the input data
#     features = np.array(features).reshape(1, 8, -1)
#     features = model.transform(features)

#     # Make a prediction
#     prediction = model.predict(features)

#     # Format the prediction as needed
#     result = {'prediction': prediction[0, 0]}

#     return render_template('result.html', result=result)

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask,render_template
# from flask import request
# import pickle
# import numpy as np
# filename='model.pkl'
# cls=pickle.load(open(filename,'rb'))

# app=Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')
# @app.route('/predict',methods=['POST'])
# # def predict():
# #     # Get form data and perform prediction
# #     # ...
# #     return render_template('results.html', result=my_prediction)
# def predict():
#     if request.method=='POST':
#         # age = int(request.form['age'])
#         # sex = int(request.form['sex'])
#         # on_thyroxine = int(request.form['on_thyroxine'])
#         # on_antithyroid_medication = int(request.form['on_antithyroid_medication'])
#         # hypopituitary = int(request.form['hypopituitary'])
#         # psych = int(request.form['psych'])
#         # goitre = int(request.form['goitre'])
#         # TSH = (request.form['TSH'])
#         # T3_measured=int(request.form['T3_measured'])
#         # TT4 = int(request.form['TT4'])
#         # referral_source =int(request.form['referral_source'])
#         # FTI = int(request.form['FTI'])
#         Do=float(request.form['do'])
#         Temp=float(request.form['temp'])
#         Ph=float(request.form['ph'])
#         Turbidity=int(request.form['turbidity'])
#         Conductivity=int(request.form['conductivity'])
#         Bod=float(request.form['bod'])
#         Nitrate=float(request.form['nitrate'])
#         Coliform=int(request.form['coliform'])
        


#         # Make a prediction
#         data= np.array([[Do, Temp, Ph,Turbidity,Conductivity,Bod,Nitrate,Coliform]])
#         my_prediction=cls.predict(data)

#     return render_template('result.html',prediction=my_prediction)
# if __name__=='__main__':
#     app.run(debug=True)
from flask import Flask,render_template
from flask import request
import numpy as np
import joblib



app=Flask(__name__)

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/predict',methods=['POST'])
# def predict():
#     # Get form data and perform prediction
#     # ...
#     return render_template('results.html', result=my_prediction)
def predict():
    if request.method=='POST':
        Do = float(request.form['D.O. (mg/l)'])
        Temp = float(request.form['Temp'])
        Ph= float(request.form['PH'])
        Turbidity = float(request.form['Turbidity (NTU)'])
        Conductivity= float(request.form['CONDUCTIVITY (µmhos/cm)'])
        Bod = float(request.form['B.O.D. (mg/l)'])
        Nitrate= float(request.form['NITRATENAN N+ NITRITENANN (mg/l)'])
        Coliform=float(request.form['TOTAL COLIFORM (MPN/100ml)Mean'])


        # Make a prediction
        l=[]
        for i in [Do,Temp,Ph,Turbidity,Conductivity,Bod,Nitrate,Coliform]:
            l.append(float(i))
        data= np.array([l])
        print(data)
        my_prediction=classifier.predict(data)

    return render_template('result.html',prediction=my_prediction)
if __name__=='__main__':
    classifier=joblib.load('final.pkl')
    app.run(debug=True)

