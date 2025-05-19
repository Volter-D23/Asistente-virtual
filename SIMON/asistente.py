import os
import time
import webbrowser
import requests
from datetime import datetime
import speech_recognition as sr
from flask import Flask, render_template, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import cohere
import pyttsx3
import random

# -------------------- CONFIGURACIÓN --------------------
COHERE_API_KEY = "DVQBpEH1uwsEMWRHWPHsb7DOGZYE9fsb9NFE9RSK"
co_client = cohere.Client(COHERE_API_KEY)

# -------------------- INTELIGENCIA ARTIFICIAL BÁSICA --------------------
intenciones = [
    "avanza", "muévete hacia adelante", "sigue derecho",
    "retrocede", "ve hacia atrás", "camina en reversa",
    "gira", "da la vuelta", "rota",
    "enciende la luz", "prende la luz",
    "abre la puerta", "desbloquea la puerta",
    "reproduce música", "pon una canción", "quiero escuchar música",
    "salir", "apágate", "terminar",
    "qué hora es", "dime la hora", "hora actual",
    "qué día es hoy", "fecha actual",
    "hola", "buenos días", "hey",
    "cuéntame un chiste", "dime un chiste", "hazme reír",
    "cómo te llamas", "cuál es tu nombre", "quién eres",
]

respuestas = [
    "Avanzando", "Avanzando", "Avanzando",
    "Retrocediendo", "Retrocediendo", "Retrocediendo",
    "Girando", "Girando", "Girando",
    "Luz encendida", "Luz encendida",
    "Puerta abierta", "Puerta abierta",
    "Reproduciendo música", "Reproduciendo música", "Reproduciendo música",
    "Saliendo del programa", "Saliendo del programa", "Saliendo del programa",
    "La hora es", "La hora es", "La hora es",
    "Hoy es", "Hoy es",
    "Hola, ¿cómo estás?", "Hola, ¿cómo estás?", "Hola, ¿cómo estás?",
    "Aquí va un chiste", "Aquí va un chiste", "Aquí va un chiste",
    "Soy SIMON, tu asistente virtual. Espero ayudarte con tus necesidades y dudas.",
    "Soy SIMON, tu asistente virtual. ¿En qué puedo ayudarte?",
    "Mi nombre es SIMON. Estoy aquí para ayudarte."
]

acciones = [
    "avanzar", "avanzar", "avanzar",
    "retroceder", "retroceder", "retroceder",
    "girar", "girar", "girar",
    "encender_luz", "encender_luz",
    "abrir_puerta", "abrir_puerta",
    "musica", "musica", "musica",
    "salir", "salir", "salir",
    "hora", "hora", "hora",
    "fecha", "fecha",
    "saludo", "saludo", "saludo",
    "chiste", "chiste", "chiste",
    "presentacion", "presentacion", "presentacion"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(intenciones)
clf = SVC()
clf.fit(X, acciones)

# -------------------- VOZ OFFLINE --------------------
def hablar(texto):
    print(f"🤖 {texto}")
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)

        voices = engine.getProperty('voices')
        for voice in voices:
            if 'spanish' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break

        engine.say(texto)
        engine.runAndWait()
    except Exception as e:
        print("❌ Error al usar pyttsx3:", e)

# -------------------- FUNCIONES --------------------

def reproducir_musica():
    canciones = [
        "https://www.youtube.com/watch?v=YROfN6pUS08",
        "https://www.youtube.com/watch?v=-vOgEO13FYA"
    ]
    webbrowser.open(random.choice(canciones))

def obtener_respuesta_chatgpt(pregunta):
    try:
        prompt = f"Responde de forma breve y concisa: {pregunta}"
        response = co_client.chat(model="command-r", message=prompt)
        return response.text.strip()
    except Exception as e:
        return f"Lo siento, ocurrió un error con Cohere: {e}"


def procesar_comando_voz():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Escuchando...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        comando = r.recognize_google(audio, language="es-ES")
        print(f"🗣️ Has dicho: {comando}")
        return procesar_texto(comando)
    except sr.UnknownValueError:
        respuesta = "No entendí lo que dijiste. Intenta de nuevo."
    except sr.RequestError:
        respuesta = "Error al conectar con el servicio de reconocimiento."
    hablar(respuesta)
    return respuesta

def procesar_texto(comando):
    entrada = vectorizer.transform([comando])
    similitudes = (entrada * X.T).toarray()[0]
    max_similitud = max(similitudes)

    umbral_similitud = 0.7

    if max_similitud < umbral_similitud:
        respuesta = obtener_respuesta_chatgpt(comando)
        hablar(respuesta)
        return respuesta

    intento = clf.predict(entrada)[0]

    if intento == "avanzar":
        respuesta = "Avanzando"
    elif intento == "retroceder":
        respuesta = "Retrocediendo"
    elif intento == "girar":
        respuesta = "Girando"
    elif intento == "encender_luz":
        respuesta = "Luz encendida"
    elif intento == "abrir_puerta":
        respuesta = "Puerta abierta"
    elif intento == "musica":
        respuesta = "Reproduciendo música"
        reproducir_musica()
    elif intento == "hora":
        respuesta = f"La hora es {datetime.now().strftime('%H:%M')}"
    elif intento == "fecha":
        respuesta = f"Hoy es {datetime.now().strftime('%d de %B de %Y')}"
    elif intento == "saludo":
        respuesta = "Hola, ¿en qué puedo ayudarte?"
    elif intento == "salir":
        respuesta = "Saliendo del programa. ¡Hasta luego!"
    elif intento == "chiste":
        respuesta = obtener_respuesta_chatgpt("Cuéntame un chiste")
    elif intento == "presentacion":
        respuesta = random.choice([
            "Soy SIMON, tu asistente virtual. Espero ayudarte con tus necesidades y dudas.",
            "Mi nombre es SIMON. Estoy aquí para ayudarte.",
            "Hola, me llamo SIMON ¿Qué necesitas?"
        ])
    else:
        respuesta = obtener_respuesta_chatgpt(comando)

    hablar(respuesta)
    return respuesta

# -------------------- FLASK --------------------
app = Flask(__name__)
historial_conversacion = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/activar', methods=['POST'])
def activar_asistente():
    respuesta = procesar_comando_voz()
    historial_conversacion.append(("usuario", "Comando de voz"))
    historial_conversacion.append(("asistente", respuesta))
    return jsonify({"respuesta": respuesta})

@app.route('/comando', methods=['POST'])
def comando():
    comando_texto = request.json.get('comando', '')
    if comando_texto:
        respuesta = procesar_texto(comando_texto)
        historial_conversacion.append(("usuario", comando_texto))
        historial_conversacion.append(("asistente", respuesta))
        return jsonify({'usuario': comando_texto, 'asistente': respuesta})
    return jsonify({'error': 'No se recibió ningún comando'}), 400

@app.route('/historial', methods=['GET'])
def obtener_historial():
    return jsonify(historial_conversacion)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    app.run(debug=True)
