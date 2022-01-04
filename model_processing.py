import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def get_prediction(self, input, class_file=None):
        prediction = self.model.predict(input)

        if class_file is None:
            pred = np.argmax(prediction[0])
        else:
            with open(class_file, 'r') as f:
                classes = [line.strip() for line in f]
            predictions = [classes[np.argmax(y)] for y in prediction]
            #predictions = [np.argmax(y) for y in prediction]
            pred = "".join(predictions)

        tf.keras.backend.clear_session()
        print(pred)
        return pred
