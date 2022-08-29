import os
from flask import Flask

# CSV 파일 경로와 임시 파일 경로입니다.
# CSV_FILEPATH = os.path.join(os.getcwd(), __name__, 'model/data/', '2020_water_quality.csv') 
# CSV_FILEPATH = 'C:\Users\shryu\OneDrive\바탕 화면\flask_app\model\data\2020_water_quality.csv'
# pipe_FILEPATH = os.path.join(os.getcwd(), __name__, 'model/data/', 'pipe.pkl') 
# sample_FILEPATH = os.path.join(os.getcwd(), __name__, 'model/data/', 'X_test_sample.pkl') 
# df_p_FILEPATH = os.path.join(os.getcwd(), __name__, 'database/DB/', 'df_p_data.pkl') 
# map2_html_FILEPATH = os.path.join(os.getcwd(), __name__, 'templates/', 'map2.html') 
CSV_FILEPATH = 'C:/Users/shryu/OneDrive/바탕 화면/project/flask_app/model/data/df.xlsx'
pipe_FILEPATH = 'C:/Users/shryu/OneDrive/바탕 화면/project/flask_app/model/data/pipe.pkl'
sample_FILEPATH = 'C:/Users/shryu/OneDrive/바탕 화면/project/flask_app/model/data/X_test_sample.pkl'
df_p_FILEPATH = 'C:/Users/shryu/OneDrive/바탕 화면/project/flask_app/database/DB/df_p_data.pkl'
map2_html_FILEPATH = 'C:/Users/shryu/OneDrive/바탕 화면/project/flask_app/templates/map2.html'

def create_app():

    app = Flask(__name__)
    
    from flask_app.main_view import main_bp
    # from flask_app.views.model_test_view import model_test_bp

    app.register_blueprint(main_bp)
    # app.register_blueprint(model_test_bp)

    return app
    
