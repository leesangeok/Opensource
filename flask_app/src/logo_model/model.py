# import torch # type: ignore
# from diffusers import StableDiffusion3Pipeline # type: ignore
# import matplotlib.pyplot as plt  # type: ignore
# import numpy as np  # type: ignore
# from model import db


# # 로컬에 저장된 모델 불러오기
# save_path = "http://cesrv.hknu.ac.kr:8991/lab/tree/MinsuKim/stable-diffusion-3.5-model"
# pipe = StableDiffusion3Pipeline.from_pretrained(save_path, torch_dtype=torch.bfloat16).to("cuda")


# def generate_logo(user_id, prompt) :

#     img_prompt = prompt #입력 프롬프트
#     img_name = str(prompt + "ex") #저장 시 이미지 파일 이름
#     img_count = 1; #생성할 이미지 개수

#     imageList = pipe(
#     img_prompt,
#     negative_prompt = "blurry image",
#     num_images_per_prompt = img_count,
#     num_inference_steps=40,
#     guidance_scale=15,
#     height=1024,
#     width=1024,
#     ).images

#     # 이미지 저장
#     for idx, image in enumerate(imageList):  # idx를 사용하기 위해 enumerate 사용
#         image.save(f"image_{idx}.png")  # 파일 이름에 idx 추가
#         db.insert_logo_Info(user_id, f"/static/gen_images/image_{idx}.png", prompt)

        

    



# # # 이미지들을 합쳐서 보여주기
# # fig, axes = plt.subplots(1, img_count, figsize=(img_count * 5, 5))  # img_count 만큼의 서브플롯 생성
# # for idx in range(img_count):  # img_count 만큼 반복
# #     axes[idx].imshow(imageList[idx])  # 각 이미지를 서브플롯에 출력
# #     axes[idx].axis('off')  # 축 숨기기
# # plt.show()  # 이미지 표시