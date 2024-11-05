from pymongo import MongoClient
from pymongo.errors import PyMongoError
import os
import logging 

username = os.getenv("USER")
db_password = os.getenv("DB_PASSWORD")
mongo_connect = f"mongodb+srv://{username}:{db_password}@logo.gyrbc.mongodb.net/?retryWrites=true&w=majority&appName=Logo"

client = MongoClient(mongo_connect)

db = client.get_database("LogoGen")
collection = db.get_collection("users")



# 유저 로그인시 신규 유저면 DB에 유저 저장, 아니라면 return으로 로그인
def signUser(user_id) :
    if findByUserId(user_id) :
        insert_user(user_id)
    
        
    
# user_id로 유저 정보를 찾는다. 있으면 user, 없으면 None 반환
def findByUserId(user_id) :
    try :
        user = collection.find_one({"user_id" : user_id})
        
        if user is None :
            logging.info("[id= %s] user_id insert ", user_id)
            return True
        else :
            return False
    
    except PyMongoError as e:
        logging.error("[id= %s] DB Error error=%s", user_id, e)
        
        

        

    

# user_id 로 데이터를 저장한다. 
def insert_user(user_id) :
    return collection.insert_one({"user_id" : user_id})
    


    
