# Imports
import numpy as np
from flask import Flask, request, render_template
import joblib
from sklearn.preprocessing import MinMaxScaler

# Cria o objeto da app
app = Flask(__name__)

#função que realiza a previsão:
def previsao_diabetes(lista_valores_formulario):
    lista_valores_formulario = np.array(lista_valores_formulario)
    entradas = lista_valores_formulario.reshape(1,8) #transfere os valores do formulário
    #normalizando entradas:
    normaliza = MinMaxScaler() #instância a classe
    features = normaliza.fit_transform(entradas)
    #carregando o modelo para predição:
    preditor = joblib.load('./static/model/modelo.sav')
    resultado = preditor.predict(features) #faz a previsão
    return resultado[0]

# Responde as requisições para o diretório raiz (/) com index.html
@app.route("/")
def index():
    return render_template('index.html', pred = " ")

# Para as previsões usamos o método POST para enviar as variáveis de entrada ao modelo
@app.route('/predict', methods = ['POST'])
def predict():    
    if request.method == 'POST':
        lista_formulario = request.form.to_dict()
        lista_features = list(lista_formulario.values())
        lista_features = list(map(lambda n: float(n), lista_features))
        resultado = previsao_diabetes(lista_features)
        if int(resultado) == 1:
            previsao = "Sim."
        else:
            previsao = "Não."
    
    # Entrega na página web as previsões
    return render_template('resultado.html', pred = previsao)

# Função main para execução do programa
def main():
    #app.run(host = '0.0.0.0', port = 8080, debug = True)  
    app.run(debug = True)  

# Execução do programa
if __name__ == '__main__':
    main()