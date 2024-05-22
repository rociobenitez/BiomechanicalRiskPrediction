from flask import Flask
from .config import Config
from .routes import routes_bp
from .users import users_bp
from .extensions import db

def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(Config)
    db.init_app(app)

    # Registrar Blueprints
    app.register_blueprint(routes_bp)
    app.register_blueprint(users_bp, url_prefix='/users')

    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory(
            directory=app.static_folder, path='ico/favicon.ico', mimetype='image/vnd.microsoft.icon'
        )

    return app

app = create_app()