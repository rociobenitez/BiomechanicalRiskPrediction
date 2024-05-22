from flask import Blueprint, request, render_template, jsonify, send_file, Response
from models.regression_model import RegressionModel
from models.classification_model import ClassificationModel
from myapp.utils import convert_to_float, process_and_validate_form_data
from myapp.storage import update_data_in_cloud, get_storage_client
from myapp.database import User
from myapp.extensions import db
from google.cloud import storage

routes_bp = Blueprint('routes', __name__)

regModel = RegressionModel()
classModel = ClassificationModel()

DYNAMIC_FIELDS = [
    "m1 hipermovil", "thomas psoas", "thomas rf", "thomas tfl", "ober", "arco aplanado",
    "arco elevado", "m1 dfx", "m5 hipermovil", "arco transverso disminuido",
    "m1 pfx", "arco transverso aumentado", "hlf", "hl", "hr", "hav",
    "index minus", "tfi", "tfe", "tti", "tte", "ober friccion", "popliteo",
    "t hintermann", "jack normal", "jack no reconstruye", "pronacion no disponible",
    "2heel raise", "heel raise", "tibia vara proximal", "tibia vara distal",
    "rotula divergente", "rotula convergente", "rotula ascendida", "genu valgo",
    "genu varo", "genu recurvatum", "genu flexum", "lunge"
]
STATIC_FIELDS = [
    "edad", "sexo", "altura", "peso", "num calzado", "articulacion", 
    "localizacion", "lado", "pace_walk", "velocidad_walk", "step rate_walk", 
    "stride length_walk", "shock_walk", "impact gs_walk", "braking gs_walk",
    "footstrike type_walk", "pronation_excursion_walk", "contact ratio_walk",
    "total force rate_walk", "step length_walk", "pronation excursion (mp->to)_walk",
    "stance excursion (fs->mp)_walk","stance excursion (mp->to)_walk",
    "fpi_total_i", "fpi_total_d"
]
FIELDS = STATIC_FIELDS + DYNAMIC_FIELDS


@routes_bp.route("/")
def home():
    return render_template('home.html', title='Models Predictions')

@routes_bp.route('/clasificacion', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
        try:
            input_features = {
                'total force rate_walk': convert_to_float(request.form['total force rate']),
                'pace_walk': convert_to_float(request.form['pace']),
                'step rate_walk': convert_to_float(request.form['step rate']),
                'thomas tfl': convert_to_float(request.form['thomas tfl']),
                'genu valgo': convert_to_float(request.form['genu valgo']),
                'pronation excursion (mp->to)_walk': convert_to_float(request.form['pronation excursion']),
                'thomas psoas': convert_to_float(request.form['thomas psoas']),
                'impact gs_walk': convert_to_float(request.form['impact gs']),
                'tfi': convert_to_float(request.form['tfi']),
                'contact ratio_walk': convert_to_float(request.form['contact ratio']),
                'stride length_walk': convert_to_float(request.form['stride length']),
                'edad': convert_to_float(request.form['edad']),
                'index minus': convert_to_float(request.form['index minus']),
                'arco aplanado': convert_to_float(request.form['arco aplanado']),
                'stance excursion (fs->mp)_walk': convert_to_float(request.form['stance excursion'])
            }
            model = classModel
            predicted_class, max_probability, probabilities, classes = model.predict(input_features)
            probabilities *= 100  # Convertir la probabilidad a porcentaje
            return render_template('classification_result.html', predicted_class=predicted_class, max_probability=max_probability, probabilities=probabilities, classes=classes, title='Predicción de Zona de Lesión 🤕')
        except ValueError as e:
            return render_template('error.html', error=str(e), title='Error de Predicción')
    return render_template('classification_form.html', title='Predicción de Zona de Lesión 🤕')

@routes_bp.route('/regresion', methods=['GET', 'POST'])
def regression():
    if request.method == 'POST':
        try:
            # Recolectar y convertir datos.
            input_features = {
                'step length_walk': convert_to_float(request.form['step length']),
                'total force rate_walk': convert_to_float(request.form['total force rate']),
                'footstrike type_walk': convert_to_float(request.form['footstrike type']),
                'stance excursion (mp->to)_walk': convert_to_float(request.form['stance excursion']),
                'imc': convert_to_float(request.form['imc'])
            }
            # Cargar el modelo y hacer predicción
            prediction = regModel.predict(input_features)
            predicted_age = int(round(prediction[0], 0))
            return render_template('regression_result.html', predicted_age=predicted_age, title='Resultado de Predicción de Edad de Marcha 🚶🏼‍♀️')
        except ValueError as e:
            return render_template('error.html', error=str(e), title='Error de Predicción')
    return render_template('regression_form.html', title='Predicción de Edad de Marcha 🚶🏼‍♀️')

@routes_bp.route("/faqs")
def faqs():
    return render_template('faqs.html', title='Preguntas Frecuentes')

@routes_bp.route('/data/<filename>')
def get_csv(filename):
    bucket_name = 'flaskapp-resources'
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f'data/processed/{filename}')
    
    # Descargar el archivo como bytes
    try:
        content = blob.download_as_bytes()
    except google.cloud.exceptions.NotFound:
        return "File not found", 404
    
    # Crear una respuesta de Flask con el contenido del archivo
    return Response(content, mimetype='text/csv')

@routes_bp.route("/runners")
def d3js():
    return render_template('d3js.html', title='Visualización con D3.js')
    
@routes_bp.route('/status', methods=['GET'])
def server_status():
    return jsonify({"status": "active", "message": "El servidor está funcionando correctamente."}), 200

@routes_bp.route('/update-database', methods=['POST', 'GET'])
def update_database():
    if request.method == 'POST':
        form_data = request.json if request.is_json else request.form.to_dict()
        app.logger.debug("Received form data: %s", form_data)
        validated_data = process_and_validate_form_data(form_data)
        app.logger.debug("Validated data: %s", validated_data)
        validated_data = update_data_in_cloud(validated_data)
        
        try:
            # Crea un nuevo usuario con los datos validados
            new_user = User(**validated_data)
            db.session.add(new_user)
            db.session.commit()
            message = "Usuario añadido exitosamente"
            status_code = 201
        except Exception as e:
            db.session.rollback()
            message = str(e)
            status_code = 500
        
        # Enviar respuesta apropiada según el tipo de solicitud
        if request.is_json:
            return jsonify({"message": message}), status_code
        else:
            return render_template('update_result.html', title='Datos actualizados')
    return render_template('update_form.html', fields=FIELDS, dynamic_fields=DYNAMIC_FIELDS, title='Añadir registro')