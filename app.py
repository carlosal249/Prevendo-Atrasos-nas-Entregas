import flask
import pandas as pd
import joblib as jb
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, send_file, escape, request, url_for
from datetime import datetime
import datetime as dt


app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')
    
    if flask.request.method == 'POST':
        #cols = ['freight_value', 'price', 'mes_pedido', 'dia_pedido', 'estacao', 'dia_da_semana', feriado_mes']
        
        data_entrega =  flask.request.form['data-entrega']
        preco_total = flask.request.form['preco_total']
        preco_frete = flask.request.form['preco_frete']

        # Garantindo que os dados venham limpos para o Dataframe
        preco_total = float(preco_total)
        preco_frete = float(preco_frete)
        
        inputs = {'data_entrega':[data_entrega], 'preco_total':[preco_total], 'preco_frete':[preco_frete]}

        data = pd.DataFrame(inputs)

        data['data_entrega'] = pd.to_datetime(data['data_entrega'])

        # Feature Engeniring
        data['weekday'] = data['data_entrega'].dt.dayofweek

        data['mes_pedido'] = data.data_entrega.dt.month
        
        data['dia_pedido'] = data.data_entrega.dt.day

        feriados_mes = [1, 4, 5, 9, 1, 11, 11 , 12, 4, 4, 6, 2, 2]
        feriados_mes_ = data['mes_pedido'].isin(feriados_mes)
        data['feriado_mes'] = feriados_mes_ * 1
        
        data.drop('data_entrega', axis=1, inplace=True)
        data['estacao'] = 3

        model = jb.load('random_forest_94.pkl.z')

        previsao_proba = model.predict_proba(data)[:, 1]

        print('----------------')
        print(previsao_proba)
        previsao_proba = previsao_proba + 0.21
        #print(previsao_proba)
        return flask.render_template('main.html', previsao_proba=previsao_proba)
        
if __name__ == '__main__':
    app.run(debug=True) #host='0.0.0.0', debug=True 