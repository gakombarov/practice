import os
import pickle
import numpy as np


class Model:

    def __init__(self):
        model_path = os.path.join('myapp', 'model.pkl')
        with open(model_path, 'rb') as f:
            self.clf = pickle.load(f)

        mapping_path = 'emnist-balanced-mapping.txt'
        self.mapping = {}
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    self.mapping[int(parts[0])] = chr(int(parts[1]))

    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray (или список)
            Входное изображение — массив из 784 пикселей
        Returns
        -------
        pred : str
            Символ-предсказание
        '''
        x_input = (np.array(x) / 255.0).reshape(1, -1)

        label_pred = int(self.clf.predict(x_input)[0])
        return self.mapping.get(label_pred, str(label_pred))