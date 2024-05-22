import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

class Config:
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql+psycopg2://{os.environ['CLOUD_SQL_USERNAME']}:"
        f"{os.environ['CLOUD_SQL_PASSWORD']}@"
        f"{os.environ['CLOUD_SQL_HOST']}/"
        f"{os.environ['CLOUD_SQL_DATABASE_NAME']}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False