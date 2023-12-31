import os
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def home_page():
    return render_template('index.html')

mod = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
mode = pickle.load(open('classifier.pkl', 'rb'))
ferti = pickle.load(open('fertilizer.pkl', 'rb'))



@app.route('/crop', methods=['GET', 'POST'])
def crop():
    result = None

    if request.method == 'POST':
        try:
            # Extract form data
            # Corrected extraction of form data
            N = float(request.form['Nitrogen'])
            P = float(request.form['Phosphorous'])
            K = float(request.form['Pottasium'])
            temp = float(request.form['Temperature'])
            humidity = float(request.form['Humidity'])
            ph = float(request.form['Ph'])
            rainfall = float(request.form['Rainfall'])


            # Create a feature list
            feature_list = [N, P, K, temp, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            # Scale the features
            scaled_features = ms.transform(single_pred)
            final_features = sc.transform(scaled_features)

            # Make prediction
            prediction = mod.predict(final_features)

            # Crop dictionary
            crop_dict = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }

            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                result = "{} is the best crop to be cultivated right there".format(crop)
            else:
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        except Exception as e:
            result = "Error: {}".format(str(e))

    return render_template("crop.html", result=result)


@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():
    result = None

    if request.method == 'POST':
        try:
            # Extract form data
            temp = float(request.form.get('temp'))
            humi = float(request.form.get('humid'))
            mois = float(request.form.get('mois'))
            soil = float(request.form.get('soil'))
            crop = float(request.form.get('crop'))
            nitro = float(request.form.get('nitro'))
            pota = float(request.form.get('pota'))
            phosp = float(request.form.get('phos'))
            
            input_data = [temp, humi, mois, soil, crop, nitro, pota, phosp]

            # Make prediction
            fertilizer_classes = {
    'Urea': 'Urea',
    'DAP': 'Diammonium Phosphate',
    '14-35-14': '14-35-14 (a fertilizer with 14% nitrogen, 35% phosphorus, and 14% potassium)',
    '28-28': '28-28 (a fertilizer with 28% nitrogen and 28% phosphorus)',
    '17-17-17': '17-17-17 (a fertilizer with 17% nitrogen, 17% phosphorus, and 17% potassium)',
    '20-20': '20-20 (a fertilizer with 20% nitrogen and 20% phosphorus)',
    '10-26-26': '10-26-26 (a fertilizer with 10% nitrogen, 26% phosphorus, and 26% potassium)'
}
            
            
            res = ferti.classes_[mode.predict([input_data])]
            res_tuple = tuple(res)

            if len(res_tuple) > 0 and res_tuple[0] in fertilizer_classes:
                result = 'Predicted Fertilizer is {}'.format(fertilizer_classes[res_tuple[0]])
            else:
                result = 'Error: Fertilizer class not found for {}'.format(res_tuple)

        except Exception as e:
            result = "Error: {}".format(str(e))

    return render_template("fertilizer.html", result=result)


@app.route('/cropping_techniques', methods=['GET', 'POST'])
def cropping_techniques():
    
    
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        crop_name = str(request.form['cropname'])

        # ph = float(request.form['ph'])

        df = pd.read_csv('C:\\Users\\bayya\\OneDrive\\Desktop\\saikiranbls\\fe.csv')

        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]
        if max_value == "N":
            if n < 0:
                key = 'NHigh'
            else:
                key = "Nlow"
        elif max_value == "P":
            if p < 0:
                key = 'PHigh'
            else:
                key = "Plow"
        else:
            if k < 0:
                key = 'KHigh'
            else:
                key = "Klow"

        fertilizer_dic = {
    'NHigh': """<p>The nitrogen (N) value of your soil is high, which might give rise to weeds. Consider the following suggestions:</p>
                <ol>
                    <li>Add manure: Manure is a simple way to amend your soil with nitrogen. Be cautious as different types of manures have varying nitrogen levels.</li>
                    <li>Coffee grounds: Use coffee grounds as a green compost material rich in nitrogen. As the grounds break down, your soil will be enriched with nitrogen. Additionally, coffee grounds improve drainage.</li>
                    <li>Plant nitrogen-fixing plants: Vegetables in the Fabaceae family, such as peas, beans, and soybeans, can increase nitrogen in your soil.</li>
                    <li>Plant 'green manure' crops: Consider crops like cabbage, corn, and broccoli as green manure to address high nitrogen levels.</li>
                    <li>Use mulch (wet grass): Mulch, including wet grass, sawdust, and soft woods, can help manage nitrogen levels while growing crops.</li>
                </ol>""",

    'Nlow': """<p>The nitrogen (N) value of your soil is low. Consider the following suggestions:</p>
                <ol>
                    <li>Add sawdust or fine woodchips: The carbon in sawdust/woodchips loves nitrogen and helps absorb excess nitrogen.</li>
                    <li>Plant heavy nitrogen-feeding plants: Tomatoes, corn, broccoli, cabbage, and spinach thrive off nitrogen and can help reduce nitrogen levels in the soil.</li>
                    <li>Water your soil: Soaking your soil with water will help leach nitrogen deeper into the soil, leaving less for plants to use.</li>
                    <li>Sugar: Limited studies suggest that adding sugar to your soil can potentially reduce nitrogen levels. Sugar, being partially composed of carbon, attracts and soaks up nitrogen.</li>
                    <li>Add composted manure: Composted manure is an excellent source of nitrogen for your plants.</li>
                    <li>Plant nitrogen-fixing plants: Consider planting peas or beans to increase nitrogen levels.</li>
                    <li>Use NPK fertilizers with high N value.</li>
                    <li>Do nothing: If you have plants producing lots of foliage, letting them absorb nitrogen may amend the soil for the next crops.</li>
                </ol>""",

    'PHigh': """<p>The phosphorus (P) value of your soil is high. Consider the following suggestions:</p>
                <ol>
                    <li>Avoid adding manure: Manure often contains high levels of phosphorous. Limiting its addition can help reduce phosphorus levels.</li>
                    <li>Use only phosphorus-free fertilizer: Choose a fertilizer with no phosphorous content, such as a 10-0-10 fertilizer, to provide other key nutrients without increasing phosphorus.</li>
                    <li>Water your soil: Soaking your soil liberally will aid in driving phosphorous out of the soil.</li>
                    <li>Plant nitrogen-fixing vegetables: Choose vegetables like beans and peas to increase nitrogen without increasing phosphorous.</li>
                    <li>Use crop rotations: Implement crop rotations to decrease high phosphorous levels.</li>
                </ol>""",

    'Plow': """<p>The phosphorus (P) value of your soil is low. Consider the following suggestions:</p>
                <ol>
                    <li>Bone meal: Use bone meal, a fast-acting source rich in phosphorous.</li>
                    <li>Rock phosphate: Apply rock phosphate, a slower-acting source that needs soil conversion for plants to use phosphorous.</li>
                    <li>Phosphorus fertilizers: Apply a fertilizer with a high phosphorous content in the NPK ratio (e.g., 10-20-10).</li>
                    <li>Organic compost: Add quality organic compost to increase phosphorous content.</li>
                    <li>Manure: Manure is an excellent source of phosphorous.</li>
                    <li>Clay soil: Introduce clay particles to help retain and fix phosphorus deficiencies.</li>
                    <li>Ensure proper soil pH: Maintain a pH in the 6.0 to 7.0 range for optimal phosphorus uptake.</li>
                    <li>Adjust soil pH: If soil pH is low, add lime or potassium carbonate; if high, add organic matter or acidifying fertilizers.</li>
                </ol>""",

    'KHigh': """<p>The potassium (K) value of your soil is high. Consider the following suggestions:</p>
                <ol>
                    <li>Loosen the soil: Deeply loosen the soil with a shovel and water thoroughly to dissolve water-soluble potassium. Repeat this process to reduce potassium levels.</li>
                    <li>Remove rocks: Sift through the soil and remove rocks to minimize the release of potassium from minerals in rocks.</li>
                    <li>Stop applying potassium-rich fertilizer: Use only commercial fertilizer with a '0' in the final number field, or switch to organic matter to enrich the soil.</li>
                    <li>Mix calcium sources: Mix crushed eggshells, crushed seashells, wood ash, or soft rock phosphate to add calcium and balance the soil.</li>
                    <li>Use NPK fertilizers with low K levels: Choose fertilizers with low potassium levels or opt for organic fertilizers with low NPK values.</li>
                    <li>Grow a cover crop of legumes: Legumes can fix nitrogen in the soil without increasing phosphorus or potassium levels.</li>
                </ol>""",

    'Klow': """<p>The potassium (K) value of your soil is low. Consider the following suggestions:</p>
                <ol>
                    <li>Mix in muriate of potash or sulphate of potash.</li>
                    <li>Try kelp meal or seaweed.</li>
                    <li>Try Sul-Po-Mag.</li>
                    <li>Bury banana peels: Bury banana peels an inch below the soil's surface to add potassium.</li>
                    <li>Use Potash fertilizers: Choose fertilizers with high potassium values.</li>
                </ol>"""
}



        response = str(fertilizer_dic[key])

        return render_template('cropping_techniques.html', response=response)
    
    else:
        return render_template('cropping_techniques.html', response=None)
        

 

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
