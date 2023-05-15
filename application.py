from flask import Flask,render_template,request

import sys
sys.path.append('/home/kousik/Desktop/PW_project/Diamond_price_prediction')
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.utils import load_object

# here pickle loading is not need if `PredictPipeline` class is used , but we do not want to use,
# because, if we use `PredictPipeline` class, the picke file will be loaded after user input,
# and production will be slower
preprocessor =load_object('artifacts/preprocessor.pkl') 
model = load_object('artifacts/model.pkl')

application=Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('predict.html')

@app.route('/perform_prediction',methods=['post'])
def perform_prediction():
    data=CustomData(
            carat=float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')
        )
    
    new_test_data = data.get_data_as_dataframe()

    # predict_pipeline = PredictPipeline()
    # test_data_pred = predict_pipeline.predict(new_test_data)

    new_test_data_transformed = preprocessor.transform(new_test_data)
    test_data_pred = model.predict(new_test_data_transformed)

    results = round(test_data_pred[0],2)

    return render_template('predict.html',final_result=results)


if __name__=="__main__":
     application.run(debug=True,host='0.0.0.0',port=5000)