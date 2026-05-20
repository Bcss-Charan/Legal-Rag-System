import logging

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from app.config import (
    MONGO_URI,
    DATABASE_NAME,
    COLLECTION_NAME
)

logger = logging.getLogger(__name__)

# MongoDB Client
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

# Database
db = client[DATABASE_NAME]

# Collection
collection = db[COLLECTION_NAME]


def get_all_documents():
    try:
        return list(collection.find())
    except PyMongoError:
        logger.exception("Failed to load legal documents from MongoDB")
        raise


def find_section(law_name, section_number):
    section_values = [str(section_number)]
    if str(section_number).isdigit():
        section_values.append(int(section_number))

    query = {
        "section_number": {
            "$in": section_values
        }
    }

    if law_name:
        query["law_name"] = {
            "$regex": f"^{law_name}$",
            "$options": "i"
        }

    try:
        return collection.find_one(query)
    except PyMongoError:
        logger.exception("Failed to find legal section in MongoDB")
        raise


def check_connection():
    try:
        client.admin.command("ping")
        return True
    except PyMongoError:
        logger.exception("MongoDB ping failed")
        return False
