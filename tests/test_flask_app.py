import unittest
from myapp import app

class TestIngestion(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_upload(self):
        response = self.app.post('/upload', json={"key": "value"})
        self.assertEqual(response.status_code, 200)
        self.assertIn('File uploaded successfully', response.data.decode())

if __name__ == '__main__':
    unittest.main()