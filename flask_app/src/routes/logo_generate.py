import os
import requests
from decorators import login_required
from flask import Flask, Blueprint, request, redirect, jsonify,render_template, session
import logging

logo_generate = Blueprint("logo_generate", __name__)


# 로고 프롬프트를 받아서 모델에 전송
@logo_generate.route("/logoGenerate", methods=['POST'])
@login_required
def RequestLogo() :
    prompt = request.get_json()
    data = prompt['description']

    logging.info("[id=%s]전송받은 프롬프트 : %s" , session['user_id'], data)

    """
    생성 프로세스
    1. 모델에 프롬프트 전송
    2. 이미지 파일 받고 DB에 user 문서에 저장
    3. 생성된 이미지를 /generate 페이지에서 바로 띄우거나 MyPage로이동시키거나
        3-1. 다시 입력받아서 바로 재생성할 수 있게 하는 게 좋아보임
        3-2. 저장을 눌러야 DB에 넣을 수 있게 하기 or 생성될때마다 DB에 저장하기
            3-2-1. 저장을 눌렀을 때 update 되게 개발필요
    4. MyPage 진입시 모든 생성 이미지와 로고 이름을 같이해서 createData와 함께 출력
    
    """
    return data



