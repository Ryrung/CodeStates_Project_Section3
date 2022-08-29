## OPEN API를 사용하여 수질데이터 요청(Json파일로 부호화)
## 요청한 데이터를 mongoDB에 업로드

from pymongo import MongoClient
from flask_app.database.get_data import get_water_data
# from get_data import get_water_data
import time

def into_monogo_DB(w_year='2022', w_mon='01', pageNo='2', numOfRows='1'):
    HOST = "cluster3.mcgho3r.mongodb.net"
    USER = "ryrung"
    PASSWORD = "ryrung0416"
    DATABASE_NAME = "projectdatabase"
    COLLECTION_NAME = "my collection"
    MONGO_URI = f"mongodb+srv://{USER}:{PASSWORD}@{HOST}/{DATABASE_NAME}?retryWrites=true&w=majority"

    client = MongoClient(MONGO_URI)
    database = client[DATABASE_NAME]
    collection = database[COLLECTION_NAME]

    df_json = get_water_data(w_year, w_mon, pageNo, numOfRows)
    collection.insert_one(df_json)
    time.sleep(1)
    return df_json

def into_monogo_DB2(data):
    HOST = "cluster3.mcgho3r.mongodb.net"
    USER = "ryrung"
    PASSWORD = "ryrung0416"
    DATABASE_NAME = "projectdatabase"
    COLLECTION_NAME = "my collection"
    MONGO_URI = f"mongodb+srv://{USER}:{PASSWORD}@{HOST}/{DATABASE_NAME}?retryWrites=true&w=majority"

    client = MongoClient(MONGO_URI)
    database = client[DATABASE_NAME]
    collection = database[COLLECTION_NAME]
    
    collection.delete_many({})
    collection.insert_many(data)
    time.sleep(1)
    