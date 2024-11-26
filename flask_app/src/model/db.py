from pymongo import MongoClient
from pymongo.errors import PyMongoError

from flask import Response, request
from datetime import datetime
import os
import logging 
from dotenv import load_dotenv


load_dotenv("key.env")
db_username = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
mongo_connect = f"mongodb+srv://{db_username}:{db_password}@logo.gyrbc.mongodb.net/?retryWrites=true&w=majority&appName=Logo"

client = MongoClient(mongo_connect)

db = client.get_database("LogoGen")
collection = db.get_collection("users")




# 유저 로그인시 신규 유저면 DB에 유저 저장, 아니라면 return으로 로그인
def signUser(user_id) :
    if findByUserId(user_id) == None :
        insert_user(user_id)
    
        
    
# user_id로 유저 정보를 찾는다. 있으면 user, 없으면 None 반환
def findByUserId(user_id) :
    try :
        user = collection.find_one({"user_id" : user_id})
        
        if user is None :
            logging.info("[id= %s] user_id is None ", user_id)
            return None

        return user
    
    except PyMongoError as e:
        logging.error("[id= %s] findByUserId() DB Error = %s", user_id, e)
        
        

# user_id 로 데이터를 저장한다. 
def insert_user(user_id) :
    try :

        collection.insert_one({"user_id" : user_id})
        logging.info("[id= %s] user 정보 저장 성공 ", user_id)
    except PyMongoError as e:
        logging.error("[id= %s] insert_user() DB Error = %s", user_id,  e)
    

# def insert_logo_Info(user_id, logo_src, logo_name) :
#     try :
#         result = collection.update_one(
#             {"_id" : user_id},
#             {"$push" : {"logo" : [{"logo_name" : logo_name,
#                                     "logo_src" : logo_src,
#                                    "createDate" : datetime.now()}]}}
#         )

#         if result.modified_count > 0:
#             logging.info("[id= %s] logo DB 저장 성공 [logo_src = %s] [logo_name = %s]", user_id, logo_src, logo_name)
#             return True
#         else:
#             logging.error("[id= %s] logo DB 저장 실패 [logo_src = %s] [logo_name = %s]", user_id, logo_src, logo_name)
#             return False
#     except PyMongoError as e:
#         logging.error("[id= %s] [logo_src = %s] [logo_name = %s]DB Error : %s", user_id, logo_src, logo_name, e)
#         return False
    


def insert_logo_Info(user_id, logo_src, prompt) :
    try :
        doc = findByUserId(user_id)
        print(doc) 

        print(user_id , logo_src, datetime.now())
        result = collection.update_one(
            {"user_id" : user_id},
            {"$push" : {"logo" : {
                                    "logo_src" : logo_src,
                                    "prompt" : prompt,
                                   "createDate" : datetime.now()}}}
        )

        if result.modified_count > 0:
            logging.info("[id= %s] logo DB 저장 성공 [logo_src = %s]", user_id, logo_src )
            return True
        else:
            logging.error("[id= %s] logo DB 저장 실패 [logo_src = %s] ", user_id, logo_src )
            return False
        
    except PyMongoError as e:
        logging.error("[id= %s] [logo_src = %s] DB Error : %s", user_id, logo_src, e)
        return False
    


