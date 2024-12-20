from routes.kakao import kakaoLogout
from decorators import login_required
from flask import  Blueprint, request, redirect, jsonify,render_template, session
from model import db

main = Blueprint('main', __name__)

@main.route('/')
def hello():
    return render_template('main.html')

@main.route('/logout')
def logout():

    if session['user_id'] :
        kakaoLogout(session.pop('access_token'),
                    session.pop('user_id'))
    
        return redirect('/')

@main.route('/generate')
@login_required
def generateLogoImage():
    return render_template('generation_page.html')
    
@main.route('/test')
@login_required
def myPage():
    user_id = int(session['user_id'])
    user = db.findByUserId(user_id)

    logo = []
    
    # logo 존재 여부 확인
    if user.get('logo' , False) :
        # 최신 순으로 순서 바꾸기 (정렬) 
        logo = user['logo']
        logo.reverse()


    return render_template('myPage.html', userData = logo)


@main.route('/myPage')
@login_required
def test() :
    user_id = int(session['user_id'])
    user = db.findByUserId(user_id)

    logo = []
    
    # logo 존재 여부 확인
    if user.get('logo' , False) :
        # 최신 순으로 순서 바꾸기 (정렬) 
        logo = user['logo']
        logo.reverse()


    return render_template('mypage_main.html', userData = logo)


@main.route('/saveImage', methods=['POST'])
def saveImage() :
    user_id = int(request.args.get('id'))
    src = request.args.get('image_src')
    prompt = request.args.get('prompt')

    result = db.insert_logo_Info(user_id, src,prompt)

    if result :
        return jsonify("{성공}"), 200
    else : 
        return jsonify("{실패}"), 500    



