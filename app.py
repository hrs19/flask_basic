import numpy as np
#import sklearn.linear_model.base
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Pickle_SVC_Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    print(prediction)
    #output = round(prediction[0], 2)
    if prediction==0:
        output="Setosa"
    elif prediction==1:
        output="Versicolor"
    elif prediction==2:
        output="Virginica"
    

    return render_template('index.html', prediction_text=f'The species is Iris:{output}')#.format(output))




if __name__ == "__main__":
    app.run(debug=True)