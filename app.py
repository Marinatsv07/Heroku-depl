from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

app = Flask(__name__, template_folder='templates')

# Load the model
with open('Week_4/linear_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open('Week_4/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        PULocationID = request.form['PULocationID']
        DOLocationID = request.form['DOLocationID']
        print(f"PULocationID %s", PULocationID)
        print(f"DOLocationID %s", DOLocationID)

        # Create a dataframe for the input data
        input_data = pd.DataFrame([{'PULocationID': PULocationID, 'DOLocationID': DOLocationID}])
        
        # Convert location IDs to strings
        input_data['PULocationID'] = input_data['PULocationID'].astype(str)
        input_data['DOLocationID'] = input_data['DOLocationID'].astype(str)

        # Transform the input data using the loaded vectorizer
        data_dicts = input_data.to_dict(orient='records')
        X_input = vectorizer.transform(data_dicts)

        # Make predictions using the loaded model
        prediction = model.predict(X_input)
        # Return the prediction result
        prediction_text = f'Predicted Trip Duration: {prediction[0]:.2f} minutes'
    except Exception as e:
        prediction_text = f"An error occurred: {e}"

    return render_template('index.html', prediction_text=prediction_text)
    #     # Return the prediction result
    #     time = prediction[0]
    #     print(f"time =%f ", time)
    #     return render_template('index.html', prediction_text='Predicted Trip Duration: {:.2f} minutes'.format(time))

    # except Exception as e:
    #     return render_template('index.html', prediction_text=f"An error occurred: {e}")

if __name__ == "__main__":
    app.run(debug=True, port=5001)