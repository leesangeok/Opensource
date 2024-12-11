# Logo Generator
Stable Diffusion 1.5 모델을 Fine-Tunning하여 프롬프트를 입력하면 적절한 Logo 이미지를 생성해주는 모델을 활용한 애플리케이션



# 프로젝트 구성 및 사용법
- Flask WAS 서버에 서버사이드렌더링으로 웹 애플리케이션 개발
- MongoDB Atlas를 이용한 NoSQL 
- 로고 생성 모델을 포함

## Flask WAS 서버 
### .env 파일 설정

- 카카오
  - CLIENT_ID = 카카오 REST_API KEY
  - REDIRECT_URI = "http://localhost:5000/oauth"
  - CLIENT_SECRET = ""


- MongoDB
  - USER=\<USERNAME>
  - DB_PASSWORD=\<PASSWORD>

- SESSION KEY
  - SECRET_KEY= 임의의 UUID 등 값 설정

#### 실행방법
```
cmd 실행
flask_app/src 이동
python app.py 로 실행
```

### Requirements
```
Flask==3.0.3
pymongo==4.10.1
python-dotenv==1.0.1
requests==2.31.0
```

# AI 모델 학습
이 모델은 [심볼(로고) 생성 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71440){:target="_blank"}데이터셋을 사용하여 학습되었습니다. 데이터셋은 총 466984장의 라벨과 이미지 쌍으로 학습 하였습니다.

모델 학습은 다음과 같은 방법으로 진행되었습니다:
- **Fine-Tuning**: 기존의 [Stable Diffusion 1.5 모델](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5){:target="_blank"}을 기반으로 하여, [특정 하이퍼파라미터]를 조정하여 학습하였습니다.
- **학습 과정**: 모델의 성능을 최적화하기 위해 `text_encoder`와 `unet`을 학습시켰습니다. 
  - **에포크 수**: 10
  - **배치 크기**: 4
  - **학습률**: 1e-6
  - **손실 함수**: Mean Squared Error (MSE) Loss
  - **최적화 알고리즘**: AdamW Optimizer
  - **GradScaler**: 자동 혼합 정밀도(Amp) 사용
  - **데이터 증강**: 이미지 크기 조정 및 정규화 적용
 

## fix
- 24.12.7 swiper 미작동현상 해결
- 24.12.6 swiper , error page 추가 및 날짜별 로고 데이터 DB 처리
- 24.12.1 sweetalert2 적용
- 24.11.24 MongoDB find 문제 해결
