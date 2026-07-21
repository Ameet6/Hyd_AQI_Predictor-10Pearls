"""
One shared way to get a MongoDB collection.
Every script (feature pipeline, backfill, training, dashboard) imports
get_collection() instead of creating its own MongoClient.
"""

from pymongo import MongoClient
from src import config


def get_client() -> MongoClient:
    return MongoClient(config.MONGO_URI)


def get_collection(collection_name: str, client: MongoClient = None):
    """
    Pass an existing client if you already have one open (recommended in
    scripts that make multiple calls), otherwise a new one is created.
    """
    if client is None:
        client = get_client()
    db = client[config.DB_NAME]
    return db[collection_name]