import csv
import os
import json
from io import StringIO
from google.cloud import storage
import google.auth
from google.oauth2 import service_account


# Definir el orden de las columnas según el archivo CSV
ordered_fields = [
    "edad", "sexo", "altura", "peso", "num calzado", "articulacion",
    "localizacion", "lado", "pace_walk", "velocidad_walk", "step rate_walk",
    "stride length_walk", "shock_walk", "impact gs_walk", "braking gs_walk",
    "footstrike type_walk", "pronation excursion_walk", "contact ratio_walk",
    "total force rate_walk", "step length_walk", "pronation excursion (mp->to)_walk",
    "stance excursion (fs->mp)_walk","stance excursion (mp->to)_walk", "m1 hipermovil",
    "thomas psoas", "thomas rf", "thomas tfl", "ober", "arco aplanado",
    "arco elevado", "m1 dfx", "m5 hipermovil", "arco transverso disminuido",
    "m1 pfx", "arco transverso aumentado", "hlf", "hl", "hr", "hav",
    "index minus", "tfi", "tfe", "tti", "tte", "ober friccion", "popliteo",
    "t hintermann", "jack normal", "jack no reconstruye", "pronacion no disponible",
    "2heel raise", "heel raise", "fpi_total_i", "fpi_total_d",
    "tibia vara proximal", "tibia vara distal", "rotula divergente",
    "rotula convergente", "rotula ascendida", "genu valgo",
    "genu varo", "genu recurvatum", "genu flexum", "lunge"
]


def load_credentials_from_gcs(bucket_name, blob_name):
    """Carga el archivo de credenciales desde Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        local_path = '/tmp/sistemas-predictivo-lesiones-8e85093137ff.json'
        blob.download_to_filename(local_path)
        print("Archivo descargado con éxito.")
        return local_path
    except Exception as e:
        print(f"Error al descargar el archivo: {e}")
        raise


def get_authenticated_session():
    """Crea una sesión autenticada con las credenciales de GCS."""
    try:
        credentials_path = load_credentials_from_gcs('flaskapp-resources', 'credentials/sistemas-predictivo-lesiones-8e85093137ff.json')
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        credentials, _ = google.auth.default()
        print("Sesión autenticada con éxito.")
        return credentials
    except Exception as e:
        print(f"Error al autenticar usando las credenciales: {e}")
        raise


def process_and_validate_form_data(form_data):
    validated_data = {}
    for field in ordered_fields:
        if field in ['articulacion', 'localizacion', 'lado']:
            validated_data[field] = form_data.get(field, '')
        else:
            try:
                validated_data[field]  = float(form_data.get(field, '0'))
            except ValueError:
                validated_data[field]  = 0.0  # Establecer a 0.0 si la conversión falla
    return validated_data


def update_data_in_cloud_storage(validated_data):
    # Calcular el IMC si es posible
    peso = float(validated_data.get('peso', 0))
    altura = float(validated_data.get('altura', 0))
    if altura > 0:  # Asegurarse de que la altura no sea cero para evitar división por cero
        altura_metros = altura / 100  # Convertir cm a metros
        validated_data['imc'] = round(peso / (altura_metros ** 2), 2)
    else:
        validated_data['imc'] = 0  # IMC no calculable

    # Añadir 'zona afectada'
    validated_data['zona afectada'] = f"{validated_data['articulacion']}_{validated_data['localizacion']}_{validated_data['lado']}"
    columns = ordered_fields + ['imc', 'zona afectada']  # Añadir campos calculados o adicionales necesarios para el almacenamiento

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, delimiter=';')
    writer.writeheader()
    writer.writerow(validated_data)

    # Subir a GCS
    client = storage.Client(credentials=get_authenticated_session())
    bucket = client.get_bucket('flaskapp-resources')
    blob = bucket.blob('data/dataset_updated.csv')
    blob.upload_from_string(output.getvalue(), 'text/csv')
    output.close()