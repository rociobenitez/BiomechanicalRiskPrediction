
import unittest
from flask_testing import TestCase
from myapp import app, create_app

class MyTest(TestCase):
    def create_app(self):
        app = create_app()
        app.config['TESTING'] = True
        return app

    def test_home(self):
        response = self.client.get('/')
        self.assert200(response)
        self.assertEqual(response.data.decode(), 'Hello, World!')


from myapp import db

class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        db.create_all()

    def test_data_ingestion(self):
        response = self.app.post('/update-database', data={
            'edad': '30', 'sexo': 'masculino', # y otros campos necesarios
        })
        self.assertEqual(response.status_code, 200)
        # Verificar que los datos están en la base de datos
        user = User.query.first()
        self.assertEqual(user.edad, 30)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

if __name__ == '__main__':
    unittest.main()
