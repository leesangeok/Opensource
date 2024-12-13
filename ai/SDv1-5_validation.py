import torch
from diffusers import DiffusionPipeline
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from dataset import CustomDataset

# GPU 설정
# torch.cuda.set_per_process_memory_fraction(0.5, device=torch.cuda.current_device())
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(0)

# 모델 로드 및 float32로 설정
modelName = "stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained("./"+modelName, torch_dtype=torch.float32)
pipe = pipe.to("cuda")


scheduler = pipe.scheduler
# 스케줄러에 timesteps 설정 (Inference steps 설정)
num_inference_steps = 150  # 원하는 inference 스텝 수로 설정하세요
scheduler.set_timesteps(num_inference_steps)

# 모델의 구성 요소 추출
unet = pipe.unet
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
vae = pipe.vae

unet.eval()
text_encoder.eval()

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 각 채널에 대해 정규화
])

validation_image_root = "/home/oss_1/data/240.심볼(로고) 생성 데이터/01-1.정식개방데이터/Validation/01.원천데이터"
validation_label_root = "/home/oss_1/data/240.심볼(로고) 생성 데이터/01-1.정식개방데이터/Validation/02.라벨링데이터"

validation_dataset = CustomDataset(
    validation_image_root, 
    validation_label_root, 
    transform=transform, 
    mode='validation'
    )
validation_dataloader = DataLoader(
    validation_dataset, 
    batch_size=16, 
    shuffle=False, 
    num_workers=16
    )


# 손실 값을 저장할 리스트
losses = []

epochs = 5

for epoch in range(epochs):
    print(f"Validation {epoch+1}/{epochs}")
    with tqdm(validation_dataloader, desc="Validation", unit="batch") as pbar:
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

            with torch.no_grad():
                # 자동 혼합 정밀도 적용
                with torch.cuda.amp.autocast():
                    # 캡션 인코딩
                    encoder_hidden_states = text_encoder(input_ids)[0]

                    # 이미지 인코딩 (VAE)
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # 노이즈 추가 및 예측
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # 손실 계산
                    loss = F.mse_loss(noise_pred, noise)

                # 손실 값을 리스트에 추가
                losses.append(loss.item())

                # 로깅
                pbar.set_postfix({'loss': loss.item()})



loss_df = pd.DataFrame(losses, columns=["loss"])
loss_df.to_csv( modelName + "validation_losses.csv", index=False)