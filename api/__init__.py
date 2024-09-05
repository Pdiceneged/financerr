from flask import Flask

app = Flask(__name__)

from api import quadro  # Importando o quadro.py que contém as rotas
