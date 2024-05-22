from flask import Blueprint, request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from myapp.database import User
from myapp.extensions import db
from myapp.utils import process_and_validate_form_data

users_bp = Blueprint('users', __name__)

@users_bp.route('/add-user', methods=['POST'])
def add_user():
    form_data = request.json
    if not form_data:
        return jsonify({"error": "No data received"}), 400

    try:
        validated_data = process_and_validate_form_data(form_data)
        new_user = User(**validated_data)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"message": "Usuario añadido exitosamente"}), 201
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@users_bp.route('/view-users', methods=['GET'])
def view_users():
    users = User.query.all()
    return jsonify({'users': [{ 
        'id': user.id, 
        'edad': user.edad, 
        'altura': user.altura, 
        'peso': user.peso,
        'num_calzado': user.num_calzado,
        'articulacion': user.articulacion,
        'localizacion': user.localizacion,
        'lado': user.lado
    } for user in users]}), 200