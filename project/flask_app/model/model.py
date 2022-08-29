import os, csv, time, shap, pickle, folium
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import OrdinalEncoder
# from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
# from flask_app import CSV_FILEPATH, pipe_FILEPATH, sample_FILEPATH, map2_html_FILEPATH
# CSV_FILEPATH = os.path.join(os.getcwd(), __name__, 'model/data/', '2020_water_quality.csv') 
CSV_FILEPATH = 'C:/Users/shryu/OneDrive/바탕 화면/project/flask_app/model/data/df.xlsx'
pipe_FILEPATH = 'C:/Users/shryu/OneDrive/바탕 화면/project/flask_app/model/data/pipe.pkl'
sample_FILEPATH = 'C:/Users/shryu/OneDrive/바탕 화면/project/flask_app/model/data/X_test_sample.pkl'
df_p_FILEPATH = 'C:/Users/shryu/OneDrive/바탕 화면/project/flask_app/database/DB/df_p_data.pkl'
map2_html_FILEPATH = 'C:/Users/shryu/OneDrive/바탕 화면/project/flask_app/templates/map2.html'

## csv to dataframe in data folder
def load_csv():
    df = pd.read_excel(CSV_FILEPATH, names=['년도','월','회차','수질측정망_명','수질측정망_코드','검사일자','항목코드'
                                      ,'항목명','값','항목정제여부','위도','경도','cat_id','cat_did'])
    return df


def EDA(df):
    df = df.pivot_table(index=['년도','월','회차','검사일자','수질측정망_코드','수질측정망_명','위도','경도'], columns='항목명', values='값')
    df.reset_index(inplace=True)
    return df


## 모델링용 feature engineering
def feat_engi(df):
    ## 등급에서 '삭제' 대상인 것들을 추림 (등급을 확인할 수 없는 항은 제거)
    ## 음용가능/불가능 이진 분류로 타겟 정리
    ## Target에 영향을 준 feature 제거 (정보누수 차단)
    conditionlist = [
        (df['수소이온농도(pH)'] >= 6.5) & (df['생물학적산소요구량(BOD)'] <= 1) & (df['화학적산소요구량(COD)'] <= 2) & (df['총유기탄소(TOC)'] <= 2) & 
        (df['부유물질(SS)'] <= 22) & (df['용존산소(DO)'] >= 7.5) & (df['총인(T-P)'] <= 0.02),
        (df['수소이온농도(pH)'] >= 6.5) & (df['생물학적산소요구량(BOD)'] <= 2) & (df['화학적산소요구량(COD)'] <= 4) & (df['총유기탄소(TOC)'] <= 3) & 
        (df['부유물질(SS)'] <= 25) & (df['용존산소(DO)'] >= 5) & (df['총인(T-P)'] <= 0.04),
        (df['수소이온농도(pH)'] >= 6.5) & (df['생물학적산소요구량(BOD)'] <= 3) & (df['화학적산소요구량(COD)'] <= 5) & (df['총유기탄소(TOC)'] <= 4) & 
        (df['부유물질(SS)'] <= 25) & (df['용존산소(DO)'] >= 5) & (df['총인(T-P)'] <= 0.1),
        (df['수소이온농도(pH)'] >= 6.5) & (df['생물학적산소요구량(BOD)'] <= 5) & (df['화학적산소요구량(COD)'] <= 7) & (df['총유기탄소(TOC)'] <= 5) & 
        (df['부유물질(SS)'] <= 25) & (df['용존산소(DO)'] >= 5) & (df['총인(T-P)'] <= 0.2),
        (df['수소이온농도(pH)'] >= 6) & (df['생물학적산소요구량(BOD)'] <= 8) & (df['화학적산소요구량(COD)'] <= 9) & (df['총유기탄소(TOC)'] <= 6) & 
        (df['부유물질(SS)'] <= 100) & (df['용존산소(DO)'] >= 2) & (df['총인(T-P)'] <= 0.3),
        (df['수소이온농도(pH)'] >= 6) & (df['생물학적산소요구량(BOD)'] <= 10) & (df['화학적산소요구량(COD)'] <= 11) & (df['총유기탄소(TOC)'] <= 8) & 
        (df['용존산소(DO)'] >= 2) & (df['총인(T-P)'] <= 0.5),
        (df['생물학적산소요구량(BOD)'] > 10) & (df['화학적산소요구량(COD)'] > 11) & (df['총유기탄소(TOC)'] > 8) & 
        (df['용존산소(DO)'] < 2) & (df['총인(T-P)'] > 0.5),
        (df['수소이온농도(pH)'] < 6) | (df['수소이온농도(pH)'] > 8.5) | (df['생물학적산소요구량(BOD)'] > 10) | (df['화학적산소요구량(COD)'] > 11) |
        (df['부유물질(SS)'] > 100) | (df['용존산소(DO)'] < 2) | (df['총인(T-P)'] > 0.5)
    ]
    # Target 정리
    choicelist = ['매우좋음', '좋음', '약간좋음', '보통', '약간나쁨', '나쁨', '매우나쁨', '매우나쁨']
    df['등급'] = np.select(conditionlist, choicelist, default='삭제')

    # 매우좋음, 좋음, 약간좋음은 음용가능으로 판단(이진분류)
    df['음용가능'] = (df['등급'] == '매우좋음') | (df['등급'] == '약간좋음') | (df['등급'] == '좋음')
    
    # 필요없는 feature 제거
    drop_idx = df[df['등급']=='삭제'].index
    df = df.drop(drop_idx)
    df.drop(['등급'],axis=1,inplace=True)
    df.drop(['수소이온농도(pH)','생물학적산소요구량(BOD)','화학적산소요구량(COD)','총유기탄소(TOC)',
             '부유물질(SS)','용존산소(DO)','총인(T-P)','클로로필-a(Chlorophyll-a)','용존총인(DTP)'],axis=1,inplace=True) 
    return df


## 디스플레이용
def feat_engi2(df):
    ## 등급에서 '삭제' 대상인 것들을 추림 (등급을 확인할 수 없는 항은 제거)
    ## 음용가능/불가능 이진 분류로 타겟 정리
    ## Target에 영향을 준 feature 제거 (정보누수 차단)
    conditionlist = [
        (df['수소이온농도(pH)'] >= 6.5) & (df['생물학적산소요구량(BOD)'] <= 1) & (df['화학적산소요구량(COD)'] <= 2) & (df['총유기탄소(TOC)'] <= 2) & 
        (df['부유물질(SS)'] <= 22) & (df['용존산소(DO)'] >= 7.5) & (df['총인(T-P)'] <= 0.02),
        (df['수소이온농도(pH)'] >= 6.5) & (df['생물학적산소요구량(BOD)'] <= 2) & (df['화학적산소요구량(COD)'] <= 4) & (df['총유기탄소(TOC)'] <= 3) & 
        (df['부유물질(SS)'] <= 25) & (df['용존산소(DO)'] >= 5) & (df['총인(T-P)'] <= 0.04),
        (df['수소이온농도(pH)'] >= 6.5) & (df['생물학적산소요구량(BOD)'] <= 3) & (df['화학적산소요구량(COD)'] <= 5) & (df['총유기탄소(TOC)'] <= 4) & 
        (df['부유물질(SS)'] <= 25) & (df['용존산소(DO)'] >= 5) & (df['총인(T-P)'] <= 0.1),
        (df['수소이온농도(pH)'] >= 6.5) & (df['생물학적산소요구량(BOD)'] <= 5) & (df['화학적산소요구량(COD)'] <= 7) & (df['총유기탄소(TOC)'] <= 5) & 
        (df['부유물질(SS)'] <= 25) & (df['용존산소(DO)'] >= 5) & (df['총인(T-P)'] <= 0.2),
        (df['수소이온농도(pH)'] >= 6) & (df['생물학적산소요구량(BOD)'] <= 8) & (df['화학적산소요구량(COD)'] <= 9) & (df['총유기탄소(TOC)'] <= 6) & 
        (df['부유물질(SS)'] <= 100) & (df['용존산소(DO)'] >= 2) & (df['총인(T-P)'] <= 0.3),
        (df['수소이온농도(pH)'] >= 6) & (df['생물학적산소요구량(BOD)'] <= 10) & (df['화학적산소요구량(COD)'] <= 11) & (df['총유기탄소(TOC)'] <= 8) & 
        (df['용존산소(DO)'] >= 2) & (df['총인(T-P)'] <= 0.5),
        (df['생물학적산소요구량(BOD)'] > 10) & (df['화학적산소요구량(COD)'] > 11) & (df['총유기탄소(TOC)'] > 8) & 
        (df['용존산소(DO)'] < 2) & (df['총인(T-P)'] > 0.5),
        (df['수소이온농도(pH)'] < 6) | (df['수소이온농도(pH)'] > 8.5) | (df['생물학적산소요구량(BOD)'] > 10) | (df['화학적산소요구량(COD)'] > 11) |
        (df['부유물질(SS)'] > 100) | (df['용존산소(DO)'] < 2) | (df['총인(T-P)'] > 0.5)
    ]
    # Target 정리
    choicelist = ['매우좋음', '좋음', '약간좋음', '보통', '약간나쁨', '나쁨', '매우나쁨', '매우나쁨']
    df['등급'] = np.select(conditionlist, choicelist, default='알수없음')

    # 매우좋음, 좋음, 약간좋음은 음용가능으로 판단(이진분류)
    df['음용가능'] = (df['등급'] == '매우좋음') | (df['등급'] == '약간좋음') | (df['등급'] == '좋음')
    
    # 필요없는 feature 제거
    df.drop(['등급'],axis=1,inplace=True)
    df.drop(['수소이온농도(pH)','생물학적산소요구량(BOD)','화학적산소요구량(COD)','총유기탄소(TOC)',
             '부유물질(SS)','용존산소(DO)','총인(T-P)','클로로필-a(Chlorophyll-a)','용존총인(DTP)'],axis=1,inplace=True) 
    return df


## 훈련, 테스트 데이터 분리
def train_test_divide(df):
    df = feat_engi(df)
    target='음용가능'
    train, test = train_test_split(df, test_size=0.2, stratify = df[target], random_state=2)
    return train, test


## target 설정 및 테스트 데이터에서 분리
def target_split(df):
    target='음용가능'
    y = df[target]
    X = df.drop(target, axis=1)
    return X, y

# df = load_csv()
# print(df)

## 실행
# df = load_csv()
# df = EDA(df)
# train, test = train_test_divide(df)
# X_train, y_train = target_split(train)
# # X_val, y_val = target_split(val)
# X_test, y_test = target_split(test)


## 기준모델 정확도
def st_model(y_test):
    st_acc = y_test.value_counts(normalize=True) 
    return st_acc


## 실행
# st_acc = st_model(y_test) #==> 0.4815100154083205


## XGBC 기본모델
def fit_model(X_train, y_train):
    pipe = None 

    pipe = Pipeline([
    ('preprocessing', make_pipeline(OrdinalEncoder(), SimpleImputer())),
    ('XGBC', XGBClassifier(n_estimators=100, random_state=2, n_jobs=-1)) 
    ])

    pipe.fit(X_train, y_train)
    model = pipe.named_steps['XGBC']
    print("accuracy:", clf.best_score_)
    return pipe


## XGBC score
def XGBC_score(pipe, X_test, y_test):
    X_score = pipe.score(X_test, y_test)
    return X_score


## 실행
# pipe = fit_model(X_train, y_train)
# X_score = XGBC_score(pipe, X_test, y_test) #print(X_score)==> 0.8388876816318253


## pipe pickling(부호화)
# with open(pipe_FILEPATH,'wb') as pickle_file:
#     pickle.dump(pipe, pickle_file)
# with open(sample_FILEPATH,'wb') as pickle_file:
#     pickle.dump(X_test, pickle_file)


## pipe/model 복호화
pipe = None
with open(pipe_FILEPATH,'rb') as pickle_file:
   pipe = pickle.load(pickle_file)
with open(sample_FILEPATH,'rb') as pickle_file:
   X_test = pickle.load(pickle_file)


## shap force plot
def shap_value(row):

    model = pipe.named_steps['XGBC']
    explainer = shap.TreeExplainer(model)
    row_processed = pipe.named_steps['preprocessing'].transform(row)
    shap_values = explainer.shap_values(row_processed)

    #shap.initjs()
    force_plot = shap.force_plot(
        base_value=explainer.expected_value, 
        shap_values=shap_values, 
        features=row, 
        link='logit', # SHAP value를 확률로 변환해 표시합니다.
        matplotlib = False
    )
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return shap_html


## 예시
row = X_test.loc[[6539]] 
shap_v = shap_value(row)


def true_false_comp(pipe, X_test, y_test):
    ## 예측확률 획득
    X_test_transformed = pipe.named_steps['preprocessing'].transform(X_test)
    class_index = 1
    model = pipe.named_steps['XGBC']
    y_pred_proba = model.predict_proba(X_test_transformed)[:, class_index]
    
    ## 예측확률과 실제값을 통해 TP, FP, TN, FN 확인
    df_p = pd.DataFrame({
                        'pred_proba': y_pred_proba, # 예측확률 
                        'status_group': y_test # 실제값
                        })
    df_p = pd.merge(df_p, X_test, left_index=True, right_index=True, how='left')
        
    ## TP, FP, TN, FN 색깔 표현후 df_p에 추가
    conditionlist = [
        (df_p['status_group'] == True) & (df_p['pred_proba'] > 0.50),
        (df_p['status_group'] == True) & (df_p['pred_proba'] <= 0.50),
        (df_p['status_group'] == False) & (df_p['pred_proba'] > 0.50),
        (df_p['status_group'] == False) & (df_p['pred_proba'] <= 0.50)]
    choicelist = ['blue', 'green', 'red', 'yellow']
    df_p['color'] = np.select(conditionlist, choicelist, default='Not Specified')
    df_p
    return df_p


## 지도 시각화
def mapping(lat=37,long=127,zoom=7.2):
    m = folium.Map(location=[lat, long], zoom_start=zoom, tiles="Stamen Terrain")
    folium.CircleMarker([lat, long], radius=3, tooltip='현재위치',color='red', fill='red', fill_opacity = 100).add_to(m)
    popup1 = folium.LatLngPopup()
    m.add_child(popup1)
    return m

def save_map(m):
    m.save(map2_html_FILEPATH)
    return m    


## 실행
#lat = 37
#long = 127
#mapping(lat, long)


## 지도 시각화2
def mapping2(df, lat=37,long=127,zoom=7.2):
    m = folium.Map(location=[lat, long], zoom_start=zoom, tiles="Stamen Terrain")

    def map_point(data):
        for x in data :
            if ( not x['위도'] ) or ( not x['경도'] ) :
                pass
            else :
                folium.CircleMarker([x['위도'], x['경도']], radius=4, tooltip=x['status_group'],color=x['color'], 
                                    fill=x['color'], fill_opacity = 100).add_to(m)

    map_point(df)

    popup1 = folium.LatLngPopup()
    m.add_child(popup1)

    def save_map2(m):
        m.save(map2_html_FILEPATH)
        time.sleep(0.2)
        
    save_map2(m)
    return m

