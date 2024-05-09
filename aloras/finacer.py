import base64
import streamlit as st
from joblib import load

# Carregar o modelo, o vetorizador e o label encoder

vectorizer = load('aloras/vectorizer.joblib')
label_encoder = load('aloras/label_encoder.joblib')
model = load('aloras/random_forest_model.joblib')

st.set_page_config(
    page_title="Financeiro Análise",
    page_icon="🛍️"
)
@st.cache_data()
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("aloras/pdipaper5.png")
img2 = get_img_as_base64("aloras/pdiside.png")

page_bg_img = f"""
<style>
header, footer {{
    visibility: hidden !important;
}}

#MainMenu {{
    visibility: visible !important;
    color: #F44D00;
}}

[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:fundoesg4k/png;base64,{img}");
    background-size: cover; 
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:esgfundo1/png;base64,{img2}");
    background-position: center; 
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}

.stTextInput>div>div>input[type="text"] {{
    background-color: #C5D6ED; 
    color: #000; 
    border-radius: 7px; 
    border: 2px solid #000010; 
    padding: 5px; 
    width: 500; 
}}

@media (max-width: 360px) {{
    [data-testid="stAppViewContainer"] > .main, [data-testid="stSidebar"] > div:first-child {{
        background-size: auto;
    }}
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.sidebar.image("aloras/Logopdi.png", width=250)


# Código de estilo e imagem omitido para simplificação

# Título do aplicativo
st.title('Análise de Suprimentos (Financeiro)')

# Input do usuário
user_input = st.text_input("Digite o Histórico:")

# Botão para realizar a predição
if st.button('Prever Natureza'):
    # Vetorizar o input do usuário
    user_input_transformed = vectorizer.transform([user_input])

    # Fazer a predição
    prediction_encoded = model.predict(user_input_transformed)

    # Converter a predição codificada de volta para o nome da categoria
    prediction = label_encoder.inverse_transform(prediction_encoded)

    # Exibir a predição
    st.write(f'**A Natureza prevista é:** {prediction[0]}')

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido por [PedroFS](https://linktr.ee/Pedrofsf)")
