# config.py
from datetime import timedelta

class Config:
    SECRET_KEY = 'super-secret-key'
    SQLALCHEMY_DATABASE_URI = (
    "postgresql+psycopg2://postgres:password@localhost:5434/ikram"
)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = 'your-jwt-secret'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=1)
