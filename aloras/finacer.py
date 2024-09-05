from flask import Flask, render_template, request
import base64
from joblib import load

# Inicializar o aplicativo Flask
app = Flask(__name__)

# Carregar o modelo, o vetorizador e o label encoder
model = load('random_forest_model.joblib')
vectorizer = load('vectorizer.joblib')
label_encoder = load('label_encoder.joblib')

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("pdipaper5.png")
img2 = get_img_as_base64("pdipaper5.png")

# Rota principal
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['user_input']
        # Vetorizar o input do usuário
        user_input_transformed = vectorizer.transform([user_input])

        # Fazer a predição
        prediction_encoded = model.predict(user_input_transformed)

        # Converter a predição codificada de volta para o nome da categoria
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]

    return render_template('index.html', img=img, img2=img2, prediction=prediction)

# Iniciar o servidor
if __name__ == '__main__':
    app.run(debug=True)
