import torch
from diffusers import DiffusionPipeline
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import csv
from tqdm import tqdm
from dataset import CustomDataset

# GPU 설정
torch.cuda.set_per_process_memory_fraction(0.5, device=torch.cuda.current_device())
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(0)

# 모델 로드 및 float32로 설정
pipe = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5-512-finetuned-epoch3", torch_dtype=torch.float32)
pipe = pipe.to("cuda")

# 기존 스케줄러를 DDIMScheduler로 교체
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
scheduler = pipe.scheduler

# 스케줄러에 timesteps 설정 (Inference steps 설정)
num_inference_steps = 150  # 원하는 inference 스텝 수로 설정하세요
scheduler.set_timesteps(num_inference_steps)

# 모델의 구성 요소 추출
unet = pipe.unet
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
vae = pipe.vae

# 모델을 학습 모드로 설정
unet.train()
text_encoder.train()

# Optimizer 설정
optimizer = torch.optim.AdamW(
    list(unet.parameters()) + list(text_encoder.parameters()),
    lr=1e-6,
    weight_decay=1e-2
)

# AMP를 위한 GradScaler 생성
scaler = torch.cuda.amp.GradScaler()

# CSV 파일 초기화
# csv_file_path = "training512_loss_log.csv"
# if not os.path.exists(csv_file_path):
#     with open(csv_file_path, mode="w", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow(["Epoch", "Step", "Loss"])


# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 각 채널에 대해 정규화
])

# 데이터셋 및 DataLoader 생성
train_image_root = "/home/oss_1/data/240.심볼(로고) 생성 데이터/01-1.정식개방데이터/Training/01.원천데이터"
train_label_root = "/home/oss_1/data/240.심볼(로고) 생성 데이터/01-1.정식개방데이터/Training/02.라벨링데이터"

train_dataset = CustomDataset(train_image_root, train_label_root, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)


# 학습 루프 설정
epochs = 5

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    with tqdm(train_dataloader, desc="Training", unit="batch") as pbar:
        for step, (images, captions) in enumerate(pbar):
            images = images.to("cuda")

            inputs = tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )
            input_ids = inputs.input_ids.to("cuda")

            # 자동 혼합 정밀도 적용
            with torch.cuda.amp.autocast():
                # 캡션 인코딩
                encoder_hidden_states = text_encoder(input_ids)[0]

                # 이미지 인코딩 (VAE)
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # 노이즈 추가 및 예측
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # 손실 계산
                loss = F.mse_loss(noise_pred, noise)

            # 역전파 및 옵티마이저 스텝
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # 로깅
            if step % 100 == 0:

                pbar.set_postfix({'loss': loss.item()})
                # with open(csv_file_path, mode="a", newline="") as file:
                #     writer = csv.writer(file)
                #     writer.writerow([epoch+1, step, loss.item()])

        # 각 epoch 이후 모델 체크포인트 저장
        pipe.save_pretrained(f"stable-diffusion-v1-5-512-finetuned-epoch{epoch}")