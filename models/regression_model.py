import pickle
import pandas as pd

class RegressionModel:
    def __init__(self):
        print("Cargando modelo...")
        # Cargar el modelo desde un archivo
        with open('models/extraTrees_model.pkl', 'rb') as file:
            self.model = pickle.load(file)
        print("Modelo cargado:", self.model)

    def predict(self, input_features):
        print("Realizando predicción...")
        try:
            features_df = pd.DataFrame([input_features], columns=input_features.keys())
            print("DataFrame para predicción:", features_df)  # Debug
            prediction = self.model.predict(features_df)
            return prediction
        except ValueError as e:
            print(f"Error al convertir los datos de entrada: {e}")
        except Exception as e:
            print(f"Error al realizar la predicción: {e}") 