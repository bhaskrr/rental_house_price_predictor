from flask import Flask, request
from flask import render_template
import pandas as pd
import joblib
import zipfile

app = Flask(__name__)

# Path to the zip file and the target file inside the zip
zip_path = "./rf_regressor.zip"
file_to_extract = "rf_regressor.joblib"
output_dir = "./"

try:
    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract the specific file
        zip_ref.extract(file_to_extract, output_dir)
except KeyError:
    print(f"{file_to_extract} not found in {zip_path}")
except zipfile.BadZipFile:
    print("Error: The file is not a valid zip archive.")

# load the model
model = joblib.load("./rf_regressor.joblib")

# load the encoders
layout_type_encoder = joblib.load("./layout_type_encoder.joblib")
property_type_encoder = joblib.load("./property_type_encoder.joblib")
furnish_type_encoder = joblib.load("./furnish_type_encoder.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    layout_type = ['BHK', 'RK']
    property_type = [
                        {"value": "Apartment", "label": "Apartment"}, {"value": "Studio_Apartment", "label": "Studio Apartment"},
                        {"value": "Independent_House", "label": "Independent House"}, {"value": "Independent_Floor", "label": "Independent Floor"},
                        {"value": "Villa", "label": "Villa"}, {"value": "Penthouse", "label": "Penthouse"}
                     ]
    furnish_type = ['Furnished', 'Semi-Furnished', 'Unfurnished']
    
    if request.method == "GET":
        return render_template(
            "index.html",
            layout_type=layout_type,
            property_type=property_type,
            furnish_type=furnish_type,
        )
    
    if request.method == "POST":
        
        ordered_features = ["bedroom", "layout_type", "property_type", "area", "furnish_type", "bathroom"]
        
        # get the form data
        form_data = request.form.to_dict()
        
        if any(form_data) is not None:
            # * Encode the form data to feed into the model
            form_data["layout_type"] = int(layout_type_encoder.transform([[form_data["layout_type"]]])[0][0])
            form_data["property_type"] = int(property_type_encoder.transform([[form_data["property_type"]]])[0][0])
            form_data["furnish_type"] = int(furnish_type_encoder.transform([[form_data["furnish_type"]]])[0][0])
        
        # covert the feature values to an integer list
        feature_array = [int(form_data[x]) for x in ordered_features]
        
        data = pd.DataFrame([feature_array], columns=ordered_features)
        
        # get prediction from the model
        prediction = model.predict(data)
        # print(prediction)
        return render_template(
            "index.html",
            layout_type=layout_type,
            property_type=property_type,
            furnish_type=furnish_type,
            price = int(prediction[0]),
        )