import os
from dotenv import load_dotenv
from flask import Flask,  session, redirect, request, render_template
from datetime import timedelta
import logging
from routes.main import main
from routes.kakao import kakao
from routes.logo_generate import logo_generate
from functools import wraps

app = Flask(__name__)

load_dotenv("key.env")
app.secret_key = os.getenv("SECRET_KEY")

# 세션 설정 
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours= 1) # 세션 expire 1시간
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
    nickname = request.args.get('nickname')

    if user_id : 
        session['user_id'] = user_id
        session['access_token'] = access_token
        session['nickname'] = nickname
        return redirect("/")
    # 세션이 없다면 홈화면으로
    else :
        redirect("/")

@app.errorhandler(404)
def page_not_found(e) :
    return render_template('errors/404.html'), 404

@app.errorhandler(401)
def unauthorized(e) : 
    return render_template('errors/401.html'), 401

@app.errorhandler(500)
def internet_server_error(e) : 
    return render_template('errors/500.html'), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

