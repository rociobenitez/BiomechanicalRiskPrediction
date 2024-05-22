import functions_framework
from flask import jsonify, request
from my_module import process_and_validate_form_data, update_data_in_cloud_storage

@functions_framework.http
def update_database(request):
    """
    Endpoint HTTP para actualizar la base de datos con nuevos registros de datos de pacientes.

    Esta función procesa datos enviados mediante solicitudes POST, validando y formateando
    los datos antes de almacenarlos en Cloud Storage. Se espera que la solicitud contenga
    varios campos relacionados con datos biomecánicos y clínicos de pacientes,
    que son esenciales para actualizaciones en la investigación y análisis.

    Args:
        request (flask.Request): Objeto request que contiene los datos de la solicitud HTTP.
        Se espera que los datos vengan en formato form-data, donde cada campo
        representa una característica específica del paciente.

    Returns:
        Tuple[flask.Response, int]: Respuesta JSON indicando el éxito de la operación
        junto con el código de estado HTTP. Retorna 200 si la operación es exitosa,
        405 si el método no está permitido.
    
    Ejemplo de uso:
        POST /update-database
        Body (form-data):
            edad: "30"
            sexo: "1"
            altura: "170"
            ...
    
    El procesamiento incluye la conversión de tipos de datos y la verificación de la
    integridad de los mismos, asegurando que todos los campos necesarios estén presentes
    y sean válidos antes de proceder a la actualización del almacenamiento.
    """
    if request.method != 'POST':
        return jsonify({"error": "Method not allowed"}), 405

    # Asumiendo que los datos vienen en formato form-data o JSON
    if request.is_json:
        form_data = request.get_json()
    else:
        form_data = request.form.to_dict()
    print("Received form data: ", form_data)

    try:
        # Validar y procesar datos
        validated_data = process_and_validate_form_data(form_data)
        print("Validated data: ", validated_data)
        # Actualizar datos en el almacenamiento en la nube
        update_data_in_cloud_storage(validated_data)
        return jsonify({"message": "Datos recibidos y almacenados correctamente"}), 200
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"Error": "Server error", "details": str(e)}), 500