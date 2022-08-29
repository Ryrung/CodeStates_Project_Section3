from flask import Blueprint, render_template, request
from flask_app.model.model import X_test, mapping2, shap_value
from flask_app.data_processing import data_processing
from flask_app.database.into_mongoDB import into_monogo_DB2
import numpy as np
import time, json, os


main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    row = X_test.iloc[:1]
    lat = 37
    long = 127
    zoom = 7.2
    shap_html = shap_value(row)
    return render_template('water.html', shap_html = shap_html, lat=lat, long=long, zoom=zoom),200

@main_bp.route('/water', methods=['GET', 'POST'])
def input_data():
    try:
        if request.method == 'POST':
            row_t = X_test.iloc[:1]
            row = row_t.copy()
            row[:] = np.nan
            if request.form['id_1'] :
                ymd = request.form['id_1']
                row['년도'] = int(ymd[:4])
                row['월'] = int(ymd[4:6])
            else :
                row['년도'] = np.nan
                row['월'] = np.nan

            lat = float(request.form['id_2'])
            long = float(request.form['id_3'])
            
            row['위도'] = lat
            row['경도'] = long
            zoom = 12
            shap_html = shap_value(row)
            return render_template('water.html',shap_html = shap_html, lat=lat, long=long, zoom=zoom),200
    except:
        row = X_test.iloc[:1]
        lat = 37
        long = 127
        zoom = 7.2
        shap_html = shap_value(row)
    
    if request.method == 'GET':
        row = X_test.iloc[:1]
        lat = 37
        long = 127
        zoom = 7.2
        shap_html = shap_value(row)

    return render_template('water.html', shap_html = shap_html, lat=lat, long=long, zoom=zoom),200

@main_bp.route('/map', methods=['GET', 'POST'])
def map():
    if request.method == "GET":
        try :
            lat = float(request.args.get('lat'))
            long = float(request.args.get('long'))
            zoom = float(request.args.get('zoom'))
            if (not lat) or (not long) or (not zoom):
                return "위도 또는 경도를 입력해 주세요.", 400
            elif (type(lat) == float) or (type(long) == float) or (type(zoom)==float):
                pass
            else :
                return "숫자만 입력해 주세요.", 400
        except :
            return "정확한 값을 넣어주세요..", 400
        #zoom = 8
        #lat=37
        #long=127
        #mapping(lat, long, zoom)
        return render_template('map.html', lat=lat, long=long, zoom=zoom),200
    
    if request.method == "POST":
        loc_json = request.get_json()
        try :
            lat = loc_json['lat']
            long = loc_json['long']
            zoom = loc_json['zoom']
            return render_template('map.html', lat=lat, long=long, zoom=zoom),200
        except :
            return "정확한 값을 넣어주세요", 400

@main_bp.route('/model', methods=['GET','POST'])
def model():
    if request.method == 'GET':
        # with open(df_p_FILEPATH,'rb') as pickle_file:
        #     df_p = pickle.load(pickle_file)
        #df_json=df_p.to_json(orient='records',force_ascii=False)
        #df_temp = json.loads(df_json)

        return render_template('model_test.html'),200

    if request.method == 'POST':
        #w_year = request.form['id_4']
        #w_mon = request.form['id_5']
        #pageNo = request.form['id_6']
        #numOfRows = request.form['id_7']
        return render_template('model_test.html'),200
    
    
@main_bp.route('/map2', methods=['GET','POST'])
def map2():
    if request.method == "POST":
        w_year = request.form['id_4']
        w_mon = request.form['id_5']
        pageNo = request.form['id_6']
        numOfRows = request.form['id_7']

        df_p = data_processing(w_year, w_mon, pageNo, numOfRows)
        df_p_json=df_p.to_json(orient='records',force_ascii=False)
        df_p_temp = json.loads(df_p_json)

        i=0
        for x in df_p_temp:
            df_p_temp[i]['location'] = { 'type' : "Point", 'coordinates' : [x['위도'], x['경도']] }
            i+=1

        mongoDB_temp = {}
        i = 1
        for row in df_p_temp:
            mongoDB_temp[f'{i}'] = row
            i += 1

        # Json 파일저장
        Json_FILEPATH = os.path.join(os.getcwd(), 'flask_app\\database\\DB\\', f'df_json_{w_year}_{w_mon}_{pageNo}_{numOfRows}.json') 
        with open(Json_FILEPATH, 'w') as json_file:
            json.dump(df_p_temp, json_file)

        into_monogo_DB2(df_p_temp)
        mapping2(df_p_temp)

        time.sleep(2)
        return render_template('map2.html'), 200

    elif request.method == 'GET':
        print(1)
        return render_template('map2.html'), 200    
    
    else :
        return 'wrong~!',400


## mongoDB 대시보드
@main_bp.route('/dashboard')
def dashboard():
    
    return render_template('dashboard.html'),200