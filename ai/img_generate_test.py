import torch
from diffusers import DiffusionPipeline
import os
from datetime import datetime
import sys

# GPU 설정: CUDA_VISIBLE_DEVICES를 먼저 설정합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(0)  # 여기서 0은 CUDA_VISIBLE_DEVICES에서 지정한 첫 번째 GPU를 의미합니다.
sys.stdin.reconfigure(encoding='utf-8')

epoch = input("사용할 모델의 epoch 입력: ")
# 모델 로드 및 bfloat16로 설정
pipe = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5-512-finetuned-epoch"+epoch, torch_dtype=torch.bfloat16)

# 장치를 cuda:0으로 설정 (CUDA_VISIBLE_DEVICES에 따라 GPU 1에 할당됨)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipe = pipe.to(device)

# 이미지 저장 디렉토리 설정
output_dir = "./images"
os.makedirs(output_dir, exist_ok=True)  # 출력 디렉토리가 없으면 생성

# 프롬프트 입력 반복
while True:
    # 사용자로부터 프롬프트 입력받기
    user_prompt = input("이미지 생성을 위한 프롬프트를 입력하세요 (종료하려면 'exit' 입력): ")
    
    # 'exit' 입력 시 루프 종료
    if user_prompt.lower() == 'exit':
        print("이미지 생성기를 종료합니다.")
        break
    
    # 이미지 생성
    image = pipe(
        user_prompt,
        num_inference_steps=40,
        guidance_scale=4.5,
    ).images[0]
    
    # 이미지 저장 경로 및 파일명 설정
    file_name = f"test_{user_prompt[:20].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"  # 프롬프트의 앞 20자를 파일명에 추가
    output_path = os.path.join(output_dir, file_name)
    
    # 이미지 저장
    image.save(output_path)
    print(f"이미지가 저장되었습니다: {output_path}")