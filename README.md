# Logo Generator
Stable Diffusion 3.5을 Fine-Tunning하여 원하는 로고 prompt를 입력하면 적절한 Logo를 생성해주는 모델을 활용한 애플리케이션



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


### Requirements
```
Flask==3.0.3
pymongo==4.10.1
python-dotenv==1.0.1
requests==2.31.0
```