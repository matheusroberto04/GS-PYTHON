import cv2
import numpy as np
import os
from flask import Flask, render_template, request
from keras.models import load_model
import base64

app = Flask(__name__)
diretorio = os.path.dirname(os.path.abspath(__file__))

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Variavel para o path e carregando o modelo
model_path = os.path.join(diretorio,"keras_model.h5")
model = load_model(model_path, compile=False)

# Fazendo as  labels
labels_path = os.path.join(diretorio,"labels.txt")
class_names = open(labels_path, "r").readlines()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    class_name = None
    confidence_score = None
    encoded_image = None

    if request.method =='POST':
        if 'file' not in request.files:
            return render_template('basico.html', message='Arquivo nao encontrado')
        file = request.files['file']
        
        if file.filename == '':
            return render_template('basico.html', message='Arquivo nao encontrado')
        
        try:
            encoded_image = base64.b64encode(file.read()).decode('UTF-8')
            #Previsao
            image = cv2.imdecode(np.fromstring(base64.b64decode(encoded_image), dtype=np.uint8),cv2.IMREAD_COLOR)
            #Verificacao de imagem
            if image is None or image.size == 0:
                return render_template('basico.html',
                message='Sem imagem')
            #Arrumando a imagem para 224x224 pixels
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            image = (image / 127.5) - 1
            prediction = model.predict(image)
            index = np.argmax(prediction)
            class_name = class_names[index][2:]
            confidence_score = (prediction[0][index]*100)
            # Impressões adicionadas
            print("Class:", class_name[2:], end="")
            print("Confidence Score:", str(np.round(confidence_score)) + "%")
        except Exception as e:
            return render_template('basico.html',message=f'Erro ao carregar a imagem:{str(e)}')
    return render_template('basico.html',
    class_name=class_name,
    confidence_score=confidence_score,
    encoded_image=encoded_image)
if __name__ == '__main__':
    app.run(debug=True)