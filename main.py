from flask import Flask, render_template, request
from model_processing import Model
from image_processing import ImageProcessing
from tts_api import TTSHandler
from PIL import Image
from difflib import SequenceMatcher
import base64
import io
import csv
import tensorflow as tf

DETECT_PATH = 'models/detect_dialect.h5'
#MODEL_PATH = ['models/hanunuo_model_vgg16.h5', 'models/tagalog_model_vgg16.h5', 'models/tagbanwa_model_vgg16.h5']
MODEL_PATH = ['models/hanunuo_model.h5', 'models/tagalog_model.h5', 'models/tagbanwa_model.h5']
CLASS_PATH = ['hanunuo_classes.txt', 'tagalog_classes.txt', 'tagbanwa_classes.txt']
DIALECT = ['Hanunuo', 'Tagalog', 'Tagbanwa']
API_KEY = '4344938ca663426eab4d37f645f515a9'
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        f = request.files['img']
        img = Image.open(io.BytesIO(f.read()))
        out, dialect = process_input(img)
        i = io.BytesIO()
        img.save(i, "PNG")
        enc_img = base64.b64encode(i.getvalue())
        tts = TTSHandler(API_KEY)
        speech = tts.convert_to_speech(out)
        return render_template('display.html', img=enc_img.decode('utf-8'), out=out, tts=speech, dialect=dialect)
    else:
        return render_template('capture.html')


def process_input(input_img):
    img = ImageProcessing()

    # Comment out below if you want to test models manually
    #Convert Image to Array
    _c = img.separate_chars(input_img, chars=[], return_img=True)
    i = img.image_to_array(_c, 300, 40)

    # Comment out below if you want to test models manually
    #Detect Dialect
    dialect_detect = Model(DETECT_PATH)
    dialect = dialect_detect.get_prediction(i)

    # change dialect var to corresponding index number for MODEL_PATH and CLASS_PATH
    #OCR
    ocr = Model(MODEL_PATH[dialect])
    print(MODEL_PATH[dialect])
    chars = img.separate_chars(input_img, chars=[])

    # change machine prediction to the proper word via dictionary
    translation = ocr.get_prediction(chars, class_file=CLASS_PATH[dialect])

    if 'e' in translation or 'i' in translation or 'o' in translation or 'u' in translation or 'd' in translation or 'r' in translation:
        with open('dictionary.csv', 'rt') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                for field in row:
                    matcher = SequenceMatcher(a=field, b=translation).ratio()
                    if matcher > 0.65:
                        print(field)
                        translation = field
                        return translation, DIALECT[dialect]

    return translation, DIALECT[dialect]



if __name__ == '__main__':
    app.run(debug=True)
