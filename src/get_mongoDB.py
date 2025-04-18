import pymongo
from pymongo.database import Database
from pymongo.collection import Collection
from typing import List, Union, Optional, Dict, Any
import os
from dotenv import load_dotenv


load_dotenv()


class MongoDBClient:
    def __init__(self, mongo_url: Optional[str] = None):
        
        if mongo_url is None:
            
            mongo_url = os.environ.get("MONGO_URL")
            
        if not mongo_url:
            raise ValueError("MongoDB URI not provided and not found in environment variables")
        
        self.mongo_url: str = mongo_url
        self.client = self._connect()
        
    def _connect(self) -> Optional[pymongo.MongoClient]:
       
        try:
            client = pymongo.MongoClient(
                self.mongo_url, 
                appname="devrel.content.python", 
                connect=True
            )
            
            client.admin.command('ping')
            print("Connected to MongoDB")
            return client
        except pymongo.errors.ConnectionFailure as e:
            print(f"Connection failed: {e}")
            return None
            
    def get_database(self, db_name: str) -> Optional[Database]:
       
        if not self.client:
            print("MongoDB client not connected")
            return None
        return self.client[db_name]
    
    def get_collection(self, db_name: str, collection_name: str) -> Optional[Collection]:
        
        database = self.get_database(db_name)
        if not database:
            print(f"Database {db_name} not found")
            return None
        return database[collection_name]

   
