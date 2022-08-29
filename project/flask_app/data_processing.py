## OPEN API를 사용하여 수질데이터 요청(Json파일로 부호화)
## 요청한 데이터를 mongoDB에 업로드
## 요청한 데이터 처리 및 부호화(pickle)

from flask_app.database.get_data import get_water_data
import pandas as pd
from pandas import json_normalize
import numpy as np
import pickle, os
# from model import feat_engi2, target_split, true_false_comp
# from flask_app import pipe_FILEPATH, df_p_FILEPATH
pipe_FILEPATH = 'C:/Users/shryu/OneDrive/바탕 화면/project/flask_app/model/data/pipe.pkl'
df_p_FILEPATH = 'C:/Users/shryu/OneDrive/바탕 화면/project/flask_app/database/DB/df_p_data.pkl'

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
    ## Target 정리
    choicelist = ['매우좋음', '좋음', '약간좋음', '보통', '약간나쁨', '나쁨', '매우나쁨', '매우나쁨']
    df['등급'] = np.select(conditionlist, choicelist, default='알수없음')

    ## 매우좋음, 좋음, 약간좋음은 음용가능으로 판단(이진분류)
    df['음용가능'] = (df['등급'] == '매우좋음') | (df['등급'] == '약간좋음') | (df['등급'] == '좋음')
    
    ## 필요없는 feature 제거
    df.drop(['등급'],axis=1,inplace=True)
    df.drop(['수소이온농도(pH)','생물학적산소요구량(BOD)','화학적산소요구량(COD)','총유기탄소(TOC)',
             '부유물질(SS)','용존산소(DO)','총인(T-P)','클로로필-a(Chlorophyll-a)','용존총인(DTP)'],axis=1,inplace=True) 
    return df

def target_split(df):
    target='음용가능'
    y = df[target]
    X = df.drop(target, axis=1)
    return X, y
    
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

## json to dataframe
def data_processing(w_year='2022', w_mon='01', pageNo='1', numOfRows='1'):
    df_json = get_water_data(w_year, w_mon, pageNo, numOfRows)
    df_json_sum = []

    for i in range(len(df_json)):
        df_json_sum.extend(df_json[f'{i}']['getWaterMeasuringList']['item'])
    df = json_normalize(df_json_sum)

    ## 위/경도 도분초 => 도 변경
    def Lat_Long_tran(LAT='LAT'):
        LAT_SEC = LAT + '_SEC'
        LAT_MIN = LAT + '_MIN'
        LAT_DGR = LAT + '_DGR'

        df[LAT_SEC] = df[LAT_SEC].fillna(0)
        df[LAT_MIN] = df[LAT_MIN].fillna(0)
        df2= df['LAT_DGR'].copy()
        for i in range(len(df[LAT_DGR])):
            if not df2[i]:
                df2[i] = np.nan
            else :
                df2[i] = df[LAT_DGR][i] + df[LAT_MIN][i]/60 + df[LAT_SEC][i]/3600
        return df2

    ##실행        
    df['LAT'] = Lat_Long_tran(LAT='LAT')
    df['LON'] = Lat_Long_tran(LAT='LON')


    ## 컬럼 위치 및 이름 변경
    df = df[['WMYR', 'WMOD', 'WMWK', 'WMCYMD', 'PT_NO', 'PT_NM', 'LAT', 'LON', 'ITEM_SS', 
            'ITEM_BOD', 'ITEM_PH', 'ITEM_TEMP', 'ITEM_NH3N', 'ITEM_DOC',
            'ITEM_DTP', 'ITEM_DTN', 'ITEM_AMNT', 'ITEM_POP', 'ITEM_EC',
            'ITEM_NO3N', 'ITEM_TOC', 'ITEM_TP', 'ITEM_TN', 
            'ITEM_CLOA', 'ITEM_COD']]
    df.columns = ['년도', '월', '회차', '검사일자', '수질측정망_코드', '수질측정망_명', '위도', '경도', '부유물질(SS)',
                '생물학적산소요구량(BOD)', '수소이온농도(pH)', '수온', '암모니아성질소(NH₃-N)', '용존산소(DO)',
                '용존총인(DTP)', '용존총질소(DTN)', '유량', '인산염(PO₄-P)', '전기전도도(EC)',
                '질산성질소(NO₃-N)', '총유기탄소(TOC)', '총인(T-P)', '총질소(T-N)',
                '클로로필-a(Chlorophyll-a)', '화학적산소요구량(COD)']


    ## 문자/특수문자 제거 및 타입변환
    df['검사일자'] = df['검사일자'].str.replace('.', '')
    df['회차'] = df['회차'].str.replace('회차', '')
    for n in df.columns:
        if (n != '수질측정망_코드') and (n != '수질측정망_명') :
            df[n] = pd.to_numeric(df[n], errors='coerce')


    ## feature engineering & test target split
    df = feat_engi2(df)
    X_test, y_test = target_split(df)

    
    ## pipe 피클링
    pipe = None
    with open(pipe_FILEPATH,'rb') as pickle_file:
        pipe = pickle.load(pickle_file)
    df_p = true_false_comp(pipe, X_test, y_test)


    ## 데이터 피클링
    # df_p_FILEPATH = os.path.join(os.getcwd(), 'flask_app\\database\\DB\\', 'df_p_data.pkl') 
    with open(df_p_FILEPATH,'wb') as pickle_file:
        pickle.dump(df_p, pickle_file)
    
    return df_p