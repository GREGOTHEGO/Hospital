# Sample participant submission for testing
from flask import Flask, jsonify, request
import tensorflow as tf
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from pickle import dump
from pickle import load

app = Flask(__name__)


class Solution:
    def __init__(self):
        #Initialize any global variables here
        self.model = tf.keras.models.load_model('example.h5')
        

    def calculate_death_prob(self, timeknown, cost, reflex, sex, blood, bloodchem1, bloodchem2, temperature, race,
                             heart, psych1, glucose, psych2, dose, psych3, bp, bloodchem3, confidence, bloodchem4,
                             comorbidity, totalcost, breathing, age, sleep, dnr, bloodchem5, pdeath, meals, pain,
                             primary, psych4, disability, administratorcost, urine, diabetes, income, extraprimary,
                             bloodchem6, education, psych5, psych6, information, cancer):
        
        """
        This function should return your final prediction!
        """
        labels = ['timeknown','cost', 'reflex','sex','blood', 'bloodchem1', 'bloodchem2','temperature','race','heart','psych1','glucose','psych2','dose', 'psych3', 'bp', 'bloodchem3','confidence', 'bloodchem4', 'comorbidity', 'totalcost','breathing','age', 'sleep', 'dnr','bloodchem5','pdeath','meals','pain','primary', 'psych4','disability','administratorcost','urine','diabetes','income','extraprimary','bloodchem6','education','psych5', 'psych6','information','cancer']
        values = [x for x in [timeknown,cost,reflex,sex,blood,bloodchem1,bloodchem2,temperature,race,heart,psych1,glucose,psych2,dose,psych3,bp,bloodchem3,confidence,bloodchem4,comorbidity,totalcost,breathing,age,sleep,dnr,bloodchem5,pdeath,meals,pain,primary, psych4,disability,administratorcost,urine,diabetes,income,extraprimary,bloodchem6,education,psych5, psych6,information,cancer]]
        df = dict()
        for label, value in zip(labels, values):
            df[label] = [value]
        df = pd.DataFrame(df)
        df.replace('', 0, inplace=True)
        df.replace('male', 0, inplace=True)
        df.replace('1', 1, inplace=True)
        df.replace('M', 0, inplace=True)
        df.replace('Male', 0, inplace=True)
        df.replace('female', 1, inplace=True)
        
        df.replace('nan', 0, inplace=True)
        
        df.replace('yes', 1, inplace=True)
        df.replace('no', 0, inplace=True)
        df.replace('metastatic', -1, inplace=True)
        
        df.replace('white', 5, inplace=True)
        df.replace('black', 1, inplace=True)
        df.replace('hispanic', 2, inplace=True)
        df.replace('other', 3, inplace=True)
        df.replace('asian', 4, inplace=True)
        
        df.replace('dnr', 1, inplace=True)
        df.replace('no dnr', 4, inplace=True)
        df.replace('dnr before sadm', 2, inplace=True)
        df.replace('dnr after sadm', 3, inplace=True)
        
        df.replace('Cirrhosis', 1, inplace=True)
        df.replace('Colon Cancer', 8, inplace=True)
        df.replace('ARF/MOSF w/Sepsis', 2, inplace=True)
        df.replace('COPD', 3, inplace=True)
        df.replace('MOSF w/Malig', 4, inplace=True)
        df.replace('CHF', 5, inplace=True)
        df.replace('Lung Cancer', 6, inplace=True)
        df.replace('Coma', 7, inplace=True)
        
        df.replace('under $11k', 1, inplace=True)
        df.replace('$11-$25k', 2, inplace=True)
        df.replace('$25-$50k', 3, inplace=True)
        df.replace('>$50k', 4, inplace=True)
        
        df.replace('ARF/MOSF', 1, inplace=True)
        df.replace('Cancer', 2, inplace=True)
        df.replace('COPD/CHF/Cirrhosis', 3, inplace=True)
        
        df.replace('<2 mo. follow-up', 1, inplace=True)
        df.replace('no(M2 and SIP pres)', 2, inplace=True)
        df.replace('SIP>=30', 3, inplace=True)
        df.replace('adl>=4 (>=5 if sur)', 4, inplace=True)
        df.replace('Coma or Intub', 5, inplace=True)
        df.replace('', 0, inplace=True)
        if(df.iloc[0]['sex']==0):
            df['sex_0'] = True
            df['sex_1'] = False
        else:
            df['sex_0'] = False
            df['sex_1'] = True
        if(df.iloc[0]['cancer']==-1):
            df['cancer_-1'] = True
            df['cancer_0'] = False
            df['cancer_1'] = False
        elif(df.iloc[0]['cancer']==0):
            df['cancer_-1'] = False
            df['cancer_0'] = True
            df['cancer_1'] = False
        else:
            df['cancer_-1'] = False
            df['cancer_0'] = False
            df['cancer_1'] = True
        if(df.iloc[0]['diabetes']==0):
            df['diabetes_0.0'] = True
            df['diabetes_1.0'] = False
        else:
            df['diabetes_0.0'] = False
            df['diabetes_1.0'] = True
        
        #print(df)
        df = df.drop(columns = ['sex','diabetes','cancer'])
        df.to_csv('greg.csv')
        df.fillna(0, inplace=True)
        #print(df.shape[1])
        scale = load(open('scaler.pkl', 'rb'))
        df = scale.transform(df)
        df = np.asarray(df).astype('float32')
        prediction = self.model.predict(df)
        print(prediction[0][0])
        return float(prediction[0][0])


# BOILERPLATE
@app.route("/death_probability", methods=["POST"])
def q1():
    solution = Solution()
    data = request.get_json()
    return {
        "probability": solution.calculate_death_prob(data['timeknown'], data['cost'], data['reflex'], data['sex'], data['blood'],
                                            data['bloodchem1'], data['bloodchem2'], data['temperature'], data['race'],
                                            data['heart'], data['psych1'], data['glucose'], data['psych2'],
                                            data['dose'], data['psych3'], data['bp'], data['bloodchem3'],
                                            data['confidence'], data['bloodchem4'], data['comorbidity'],
                                            data['totalcost'], data['breathing'], data['age'], data['sleep'],
                                            data['dnr'], data['bloodchem5'], data['pdeath'], data['meals'],
                                            data['pain'], data['primary'], data['psych4'], data['disability'],
                                            data['administratorcost'], data['urine'], data['diabetes'], data['income'],
                                            data['extraprimary'], data['bloodchem6'], data['education'], data['psych5'],
                                            data['psych6'], data['information'], data['cancer'])}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)
