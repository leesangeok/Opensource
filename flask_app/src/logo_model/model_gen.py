from diffusers import DiffusionPipeline
import os
from datetime import datetime
from model import db
from flask import current_app

def generate_logo(user_id, prompt) :

    # 이미지 저장 디렉토리 설정
    output_dir = "./static/gen_images"  # 이미지 저장 디렉토리
    os.makedirs(output_dir, exist_ok=True)  # 출력 디렉토리가 없으면 생성

    # 사용자로부터 프롬프트 입력받기
    user_prompt = prompt
    pipe = current_app.config['pipe']

    # 이미지 생성
    image = pipe(
        user_prompt,
        num_inference_steps=40,
        guidance_scale=4.5,
    ).images[0]
    
    # 이미지 저장 경로 및 파일명 설정
    file_name = f"{user_prompt[:20].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"  # 프롬프트의 앞 20자를 파일명에 추가
    output_path = os.path.join(output_dir, file_name)
    
    # 이미지 저장
    image.save(output_path)

    return db.insert_logo_Info(user_id, output_path, prompt)



