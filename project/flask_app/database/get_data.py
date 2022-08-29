## OPEN API를 사용하여 수질데이터 요청(Json파일로 부호화)

import requests
import json
import time
import os, json

def get_water_data(w_year='2022', w_mon='01', pageNo='1', numOfRows='1'):
    Json_FILEPATH = os.path.join(os.getcwd(), 'flask_app\\database\\DB\\', f'df_json_{w_year}_{w_mon}_{pageNo}_{numOfRows}.json') 
    api_key = '%2BAk2G3XCsYInJPZxCbV%2FaroVBkI6kx7vaoXYlQ%2FfBQ0mMkYei9G8d%2FrDDn8xQbZ43hAiwunlG1jbiXQVQyPHLw%3D%3D'
    df_json = {}
    for i in range(0,int(pageNo)):
        url = f'http://apis.data.go.kr/1480523/WaterQualityService/getWaterMeasuringList?numOfRows={numOfRows}&pageNo={i+1}&serviceKey={api_key}&resultType=json&wmyrList={w_year}&wmodList={w_mon}'
        json_data = requests.get(url)
        df_json[f'{i}'] = json.loads(json_data.text)
        time.sleep(1)

    with open(Json_FILEPATH, 'w') as json_file:
       json.dump(df_json, json_file)
    
    return df_json

