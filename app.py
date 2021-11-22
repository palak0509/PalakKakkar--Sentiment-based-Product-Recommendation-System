from __future__ import division, print_function

import numpy as np
import pandas as pd
import joblib as joblib

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify, Markup

app = Flask(__name__)

#We need to have the 1 model for sentiment analysis and 1 recommendation system, for our analysis. 
#I've included the final processing part for the sentiment based recommendations, in this notbook (app.py), itself, thus avoiding the need of an additional model.py file. 
#We can directly call the earlier exported pkl files directly to this notebook and use it for getting our final recommendations, as below.


ratingsMatrix = joblib.load('user_rating.pkl')
productClass = joblib.load('sentiment_class.pkl')

headings = ("recommendations:")
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if (request.method == 'POST'):
        formVals = [x for x in request.form.values()]
        usrName = formVals[0].lower()
        try:
            top20 = ratingsMatrix.loc[usrName].sort_values(ascending=False)[0:20]
            for itmName in list(top20.index):
                top20[itmName] = productClass.loc[itmName][0]
            #top5 = []
            top5 = list(top20.sort_values(ascending=False)[:5].index)
            res = ""
            idx = 1
            for itm in top5:
                res += "({0}): {1} \n ".format(idx, itm)
                idx += 1

            return render_template('index.html', items_list="Top 5 recommendations for {0} are: \n {1} ".format(usrName, res))
        except Exception:
            return render_template('index.html', items_list="User doesn't exist")
    else:
        return render_template('index.html')

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)
    app.debug=True
    app.run()
