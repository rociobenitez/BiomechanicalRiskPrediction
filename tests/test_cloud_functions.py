import unittest
from unittest.mock import patch
import os
import sys
# Añade el directorio raíz al sys.path para permitir importaciones desde allí
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import update_database
from flask import Flask, request

class TestCloudFunction(unittest.TestCase):
    def test_update_database(self):
        # Datos que se enviarán a través del formulario
        form_data = {
            "edad": "31",
            "sexo": "0",
            "altura": "170",
            "peso": "60",
            "num calzado": "40",
            "articulacion": "rodilla",
            "localizacion": "medial",
            "lado": "d",
            "pace_walk": "1",
            "velocidad_walk": "1",
            "step rate_walk": "1",
            "stride length_walk": "1",
            "shock_walk": "1",
            "impact gs_walk": "1",
            "braking gs_walk": "1",
            "footstrike type_walk": "1",
            "pronation excursion_walk": "1",
            "contact ratio_walk": "1",
            "total force rate_walk": "1",
            "step length_walk": "1",
            "pronation excursion (mp->to)_walk": "1",
            "stance excursion (fs->mp)_walk": "1",
            "stance excursion (mp->to)_walk": "1",
            "fpi_total_i": "6",
            "fpi_total_d": "6",
            "m1 hipermovil": "1",
            "thomas psoas": "1",
            "thomas rf": "1",
            "thomas tfl": "1",
            "ober": "1",
            "arco aplanado": "1",
            "arco elevado": "1",
            "m1 dfx": "1",
            "m5 hipermovil": "1",
            "arco transverso disminuido": "1",
            "m1 pfx": "1",
            "arco transverso aumentado": "1",
            "hlf": "1",
            "hl": "1",
            "hr": "1",
            "hav": "1",
            "index minus": "1",
            "tfi": "1",
            "tfe": "1",
            "tti": "1",
            "tte": "1",
            "ober friccion": "1",
            "popliteo": "1",
            "t hintermann": "1",
            "jack normal": "1",
            "jack no reconstruye": "1",
            "pronacion no disponible": "1",
            "2heel raise": "1",
            "heel raise": "1",
            "tibia vara proximal": "1",
            "tibia vara distal": "1",
            "rotula divergente": "1",
            "rotula convergente": "1",
            "rotula ascendida": "1",
            "genu valgo": "1",
            "genu varo": "1",
            "genu recurvatum": "1",
            "genu flexum": "1",
            "lunge": "1"
        }
        with patch('my_module.process_and_validate_form_data') as mock_process, \
             patch('my_module.update_data_in_cloud_storage') as mock_storage:
            # Crear una instancia de la aplicación Flask y el contexto de la solicitud
            app = Flask(__name__)
            with app.test_request_context('/update-database', method='POST', data=form_data):
                mock_process.return_value = form_data  # Mock del proceso de validación
                response = update_database(request) # Ejecutar la función de endpoint con la solicitud simulada
                print(response)

if __name__ == '__main__':
    unittest.main()