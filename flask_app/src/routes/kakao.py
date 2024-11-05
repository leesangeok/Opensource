import os
import requests, logging

from flask import Flask, Blueprint, request, redirect, jsonify

kakao = Blueprint('kakao', __name__)
client_id = os.getenv('CLIENT_ID')
redirect_uri = os.getenv('REDIRECT_URI')

# 카카오로 부터 인가코드 받기
@kakao.route('/authorize')
def get_auth_code() :
    auth_url = f'https://kauth.kakao.com/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&prompt=select_account'
    # 인가코드 받는 페이지로 리다이렉트 하고 난 후 
    # 코드 발급 후 카카오 디벨롭에서 설정한 url로 리다이렉트함
    return redirect(auth_url)


# 받은 인가코드를 이용해서 Access Token 발급
@kakao.route('/oauth')
def access_token() :
    auth_code = request.args.get('code')
    
    url = f'https://kauth.kakao.com/oauth/token'
    headers = {"Content-type": "application/x-www-form-urlencoded;charset=utf-8"}
    data = {"grant_type" : "authorization_code",
            "client_id" : client_id,
            "redirect_uri" : redirect_uri,
            "code" : auth_code}
    
    #Access Token 요청
    res = requests.post(url, headers=headers, data=data)

    # Access Token 요청 실패시 로그인 페이지로 이동
    if res.status_code != 200 :
        return redirect("/login")

    # Access Token 꺼내기
    access_token = res.json()["access_token"] 
    
    # 사용자구분을 위해 id 값을 추출
    user_id = getUserInfo(access_token)["id"]

    return redirect(f"/getSession?user_id={user_id}&access_token={access_token}")
    

# Access Token을 이용해서 user 정보 가져오기
def getUserInfo(access_Token) :
    headers = {"Authorization" : f'Bearer {access_Token}',
               "Content-Type" : "application/x-www-form-urlencoded;charset=utf-8"}
    url = "https://kapi.kakao.com/v2/user/me"
    userInfo = requests.get(url, headers=headers)
    
    # 사용자 정보 가져오기 실패할 경우 로그인 페이지로 이동
    if userInfo.status_code != 200 :
        logging.info("사용자 정보 획득 실패")
        return redirect("/login")
    
    return userInfo.json()


# 카카오 로그아웃 작업
def kakaoLogout(access_token, id) :
    url = "https://kapi.kakao.com/v1/user/logout"
    headers = {"Authorization" : f"Bearer {access_token}"}
    res = requests.post(url, headers=headers)

    if res.status_code == 200 :
        logging.info("id= [%s] 로그아웃", id)
        return 

