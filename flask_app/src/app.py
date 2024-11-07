import os
from flask import Flask, render_template, session, redirect,Response, request
from datetime import timedelta
import routes.logo_generate, logging
from routes.main import main
from routes.kakao import kakao
from routes.logo_generate import logo_generate
from functools import wraps

app = Flask(__name__)


app.secret_key = os.getenv("SECRET_KEY")

# 세션 설정 
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30) # 세션 expire 30분
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True

# log 레벨을 INFO로 설정
logging.basicConfig(level=logging.INFO)

# blueprint 등록
app.register_blueprint(kakao)
app.register_blueprint(main)
app.register_blueprint(logo_generate)

@app.route('/getSession')
# 카카오 user_id로 사용자 고유id 세션 생성
def kakao_login_callback() :
    user_id = request.args.get("user_id")
    access_token = request.args.get("access_token")

    if user_id : 
        session['user_id'] = user_id
        session['access_token'] = access_token
        return redirect("/")
    
    else :
        redirect("/")





if __name__ == '__main__':
    app.run(debug=True)

