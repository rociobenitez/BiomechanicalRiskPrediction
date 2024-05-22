import pickle
import pandas as pd

class ClassificationModel:
    def __init__(self):
        print("Cargando modelo...")
        # Cargar el modelo desde un archivo
        with open('models/bagging_model.pkl', 'rb') as file:
            self.model = pickle.load(file)
        print("Modelo cargado:", self.model)

    def predict(self, input_features):
        print("Realizando predicción...")
        try:
            features_df = pd.DataFrame([input_features], columns=input_features.keys())
            print("DataFrame para predicción:", features_df)  # Debug
            predictions = self.model.predict(features_df)
            probabilities = self.model.predict_proba(features_df)
            classes = self.model.classes_  # Obtener las clases del modelo
            # Obtener la clase con la máxima probabilidad
            max_probability_index = probabilities[0].argmax()
            max_probability = probabilities[0][max_probability_index]
            predicted_class = self.model.classes_[max_probability_index]
            return predicted_class, max_probability, probabilities[0].tolist(), classes.tolist()
        except ValueError as e:
            print(f"Error al convertir los datos de entrada: {e}")
        except Exception as e:
            print(f"Error al realizar la predicción: {e}") 