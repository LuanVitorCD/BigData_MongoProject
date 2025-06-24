from dotenv import load_dotenv
import os

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "model_cache_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "models")
