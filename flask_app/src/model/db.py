from pymongo import MongoClient
from pymongo.errors import PyMongoError
from gridfs import GridFS
from flask import Response, request
import os
import logging 

username = os.getenv("USER")
db_password = os.getenv("DB_PASSWORD")
mongo_connect = f"mongodb+srv://{username}:{db_password}@logo.gyrbc.mongodb.net/?retryWrites=true&w=majority&appName=Logo"

client = MongoClient(mongo_connect)

db = client.get_database("LogoGen")
collection = db.get_collection("users")
fs = GridFS(db) 



# 유저 로그인시 신규 유저면 DB에 유저 저장, 아니라면 return으로 로그인
def signUser(user_id) :
    if findByUserId(user_id) == None :
        insert_user(user_id)
    
        
    
# user_id로 유저 정보를 찾는다. 있으면 user, 없으면 None 반환
def findByUserId(user_id) :
    try :
        user = collection.find_one({"user_id" : user_id})
        
        if user is None :
            logging.info("[id= %s] user_id insert ", user_id)
            return None

        return user
    
    except PyMongoError as e:
        logging.error("[id= %s] DB Error error=%s", user_id, e)
        
        

# user_id 로 데이터를 저장한다. 
def insert_user(user_id) :
    return collection.insert_one({"user_id" : user_id})
    

# 해당 함수는 1개의 logo_id에 대한 1개의 로고 이미지 조회
def findByImageId(logo_id):
    return fs.get(logo_id) # fs.get(ObjectId(logo_id 배열)) 이렇게 id만 담은 배열 자체를 넘겨 한번에 조회 가능 
    #logo_data =  fs.get(ObjectId(logo_id 배열))
    # logos (빈 배열) = logo_data.read() (파일 내용 읽기 함수)




# 이미지 데이터를 GridFS에 저장하고 logo_id 반환
def upload_logoImage(image_data) :
    logo = image_data
    logo_id = fs.put(logo, filename=logo.name)
    return logo_id


# DB에 Logo Image Id 값 저장 
def updateUserLogoImage(user_id, logo_id) :

    try :
        result = collection.update_one(
            {'_id': user_id},  # 조건: user_id가 일치하는 사용자 찾기
            {'$push': {'logo': [{'logo_id': logo_id}]}}  # 로고 정보 업데이트 
        )

        if result.modified_count > 0:
            logging.info("[id= %s] logo DB 저장 성공 [logo_id = %s]", user_id, logo_id)
        else:
            logging.error("[id= %s] logo DB 저장 실패 [logo_id = %s]", user_id, logo_id)

    except PyMongoError as e:
        logging.error("[id= %s] [logo_id = %s] DB Error : %s", user_id, logo_id, e)
        return



