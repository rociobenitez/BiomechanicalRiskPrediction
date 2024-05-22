import csv
import os
import json
from io import StringIO
from google.cloud import storage
import google.cloud.exceptions
from google.oauth2 import service_account

def get_storage_client():
    # Ruta al archivo de credenciales JSON en la carpeta raíz del proyecto
    credentials_path = os.path.join(os.path.dirname(__file__), '..', 'sistemas-predictivo-lesiones-8e85093137ff.json')
    if credentials_path:
        with open(credentials_path, 'r') as f:
            credentials_info = json.load(f)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        return storage.Client(credentials=credentials)
    else:
        # En App Engine, las credenciales ADC se usan automáticamente
        return storage.Client()
    
def upload_to_bucket(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def update_data_in_cloud(validated_data):
    # Calcular el IMC si es posible
    peso = float(validated_data.get('peso', 0))
    altura = float(validated_data.get('altura', 0))
    if altura > 0:  # Asegurarse de que la altura no sea cero para evitar división por cero
        altura_metros = altura / 100  # Convertir cm a metros
        validated_data['imc'] = round(peso / (altura_metros ** 2), 2)
    else:
        validated_data['imc'] = 0  # IMC no calculable

    # Añadir 'zona_afectada'
    validated_data['zona_afectada'] = f"{validated_data['articulacion']}_{validated_data['localizacion']}_{validated_data['lado']}"
    return validated_data


def update_data_in_cloud_storage(validated_data):
    validated_data = update_data_in_cloud(validated_data)

    # Definir el orden de las columnas
    columns = [
        'edad', 'sexo', 'altura', 'peso', 'num_calzado', 'articulacion', 'localizacion',
        'lado', 'pace_walk', 'velocidad_walk', 'step_rate_walk', 'stride_length_walk',
        'shock_walk', 'impact_gs_walk', 'braking_gs_walk', 'footstrike_type_walk',
        'pronation_excursion_walk', 'contact_ratio_walk', 'total_force_rate_walk',
        'step_length_walk', 'pronation_excursion_mp_to_walk', 'stance_excursion_fs_mp_walk',
        'stance_excursion_mp_to_walk', 'm1_hipermovil', 'thomas_psoas', 'thomas_rf',
        'thomas_tfl', 'ober', 'arco_aplanado', 'arco_elevado', 'm1_dfx', 'm5_hipermovil',
        'arco_transverso_disminuido', 'm1_pfx', 'arco_transverso_aumentado', 'hlf', 'hl', 'hr',
        'hav', 'index_minus', 'tfi', 'tfe', 'tti', 'tte',
        'ober_friccion', 'popliteo', 't_hintermann',
        'jack_normal', 'jack_no_reconstruye', 'pronacion_no_disponible',
        'heel_raise_double', 'heel_raise','fpi_total_i', 'fpi_total_d',
        'tibia_vara_proximal', 'tibia_vara_distal', 'rotula_divergente',
        'rotula_convergente', 'rotula_ascendida', 'genu_valgo', 'genu_varo',
        'genu_recurvatum','genu_flexum', 'lunge', 'imc', 'zona_afectada'
    ]

    # Ruta del archivo CSV en GCS
    bucket_name = 'flaskapp-resources'
    file_name = 'data/dataset_updated.csv'
    
    # Cliente de GCS
    #client = storage.Client(credentials=credentials)
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Intentar descargar el archivo existente y abrirlo en modo append
    try:
        # Descargar el archivo a un buffer
        current_data = blob.download_as_text()
        csv_output = StringIO(current_data)
        # Mover el cursor al final del archivo
        csv_output.seek(0, 2)  # Mover al final del stream
    except google.cloud.exceptions.NotFound:
        # El archivo no existe, se creará uno nuevo
        csv_output = StringIO()  # Nuevo buffer para el archivo
        writer = csv.DictWriter(csv_output, fieldnames=columns, delimiter=';')
        writer.writeheader()
    
    writer = csv.DictWriter(csv_output, fieldnames=validated_data.keys(), delimiter=';')
    writer.writerow(validated_data)

    # Subir el archivo actualizado a Cloud Storage
    blob.upload_from_string(csv_output.getvalue(), 'text/csv')
    csv_output.close()
    print("Datos añadidos con éxito al archivo CSV.")
