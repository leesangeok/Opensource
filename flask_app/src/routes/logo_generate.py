from decorators import login_required
from flask import Blueprint, request, jsonify, session,redirect
from logo_model import model_gen
import logging


logo_generate = Blueprint("logo_generate", __name__)


# 로고 프롬프트를 받아서 모델에 전송
@logo_generate.route("/logoGenerate", methods=['POST'])
@login_required
def RequestLogo() :
    data = request.get_json()
    user_id = int(session["user_id"])
    prompt = data['description']

    logging.info("[id=%s]전송받은 프롬프트 : %s" , user_id, data)

    result = model_gen.generate_logo(user_id, prompt)

    if result : 
        return jsonify(result), 200

    return data



