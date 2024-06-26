import base64
import streamlit as st
from joblib import load

# Carregar o modelo, o vetorizador e o label encoder

vectorizer = load('aloras/vectorizer.joblib')
label_encoder = load('aloras/label_encoder.joblib')




# Código de estilo e imagem omitido para simplificação

# Título do aplicativo
st.title('Análise de Suprimentos (Financeiro)')

# Input do usuário
user_input = st.text_input("Digite o Histórico:")

# Botão para realizar a predição
if st.button('Prever Natureza'):
    # Vetorizar o input do usuário
    user_input_transformed = vectorizer.transform([user_input])


    # Converter a predição codificada de volta para o nome da categoria
    prediction = label_encoder.inverse_transform(prediction_encoded)

    # Exibir a predição
    st.write(f'**A Natureza prevista é:** {prediction[0]}')

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido por [PedroFS](https://linktr.ee/Pedrofsf)")
