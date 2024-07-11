from flask import Flask,render_template,request
from src.pipeline.prediction_pipeline import PredictPipeline,CustomData

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction",methods=["GET","POST"])
def predict():
    if request.method == "GET":
        return render_template("home.html")
    else:
        form_data=CustomData(
            Age=request.form.get('Age'),
            Driving_License=request.form.get('Driving_License'),
            Region_Code=request.form.get('Region_Code'),
            Previously_Insured=request.form.get('Previously_Insured'),
            Annual_Premium=request.form.get('Annual_Premium'),
            Policy_Sales_Channel=request.form.get('Policy_Sales_Channel'),
            Vintage=request.form.get('Vintage'),
            Gender=request.form.get('Gender'),
            Vehicle_Age=request.form.get('Vehicle_Age'),
            Vehicle_Damage=request.form.get('Vehicle_Damage')
        )
    
    pred_df=form_data.get_data_as_data_frame()
    
    predict_pipeline=PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    
    if results[0]==1:
        return render_template("home.html",result="The customer is likely to buy the policy")
    else:
        return render_template("home.html",result="The customer is unlikely to buy the policy")
        

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)