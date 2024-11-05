from pymongo import MongoClient

mongo_connect = "mongodb+srv://{USER}:{DB_PASSWORD}@logo.gyrbc.mongodb.net/?retryWrites=true&w=majority&appName=Logo"

client = MongoClient(mongo_connect)