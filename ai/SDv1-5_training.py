import torch
from diffusers import DiffusionPipeline, DDIMScheduler
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import csv
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import urllib.parse
import hashlib

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# 모델 로드 및 float32로 설정
pipe = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", torch_dtype=torch.float32)
pipe = pipe.to("cuda")

# 기존 스케줄러를 DDIMScheduler로 교체
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
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
    lr=1e-6,  # 학습률을 낮춤
    weight_decay=1e-2
)

# AMP를 위한 GradScaler 생성
scaler = torch.cuda.amp.GradScaler()

# CSV 파일 초기화
csv_file_path = "training_loss_log.csv"
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Step", "Loss"])

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, image_root, label_root, transform=None):
        self.image_root = image_root
        self.label_root = label_root
        self.transform = transform
        self.image_paths = []
        self.label_paths = []

        # 이미지와 레이블 파일 경로를 일치시키는 작업
        for image_folder in os.listdir(image_root):
            if image_folder.endswith('.zip') or image_folder == '.ipynb_checkpoints':
                continue
            image_folder_path = os.path.join(image_root, image_folder)
            label_folder_path = os.path.join(label_root, f"TL_TEXT_{image_folder.split('_')[1]}_{image_folder.split('_')[2]}")

            for image_file in os.listdir(image_folder_path):
                if image_file.lower().endswith('.png'):
                    label_file = image_file.split(".")[0] + ".txt"
                    self.image_paths.append(os.path.join(image_folder_path, image_file))
                    self.label_paths.append(os.path.join(label_folder_path, label_file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 레이블 로드
        with open(label_path, "r", encoding="utf-8") as file:
            label = file.read().strip()

        return image, label

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 각 채널에 대해 정규화
])

# 데이터셋 및 DataLoader 생성
train_image_root = "/home/oss_1/data/240.심볼(로고) 생성 데이터/01-1.정식개방데이터/Training/01.원천데이터"
train_label_root = "/home/oss_1/data/240.심볼(로고) 생성 데이터/01-1.정식개방데이터/Training/02.라벨링데이터"

train_dataset = CustomDataset(train_image_root, train_label_root, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)


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

            # 로깅 및 이미지 복원
            if step % 100 == 0:
                with torch.no_grad():
                    # 첫 번째 샘플만 처리 (또는 전체 배치 처리 가능 여부에 따라)
                    idx = 0  # 또는 원하는 인덱스

                    # timestep은 스칼라 값이어야 합니다.
                    t = timesteps[idx].item()

                    # 필요한 텐서 선택
                    n_latent = noisy_latents[idx:idx+1]
                    n_pred = noise_pred[idx:idx+1]

                    # scheduler.step 호출
                    scheduler_output = scheduler.step(
                        model_output=n_pred,
                        timestep=t,
                        sample=n_latent,
                        eta=0.0
                    )
                    reconstructed_latents = scheduler_output.prev_sample

                    # VAE 디코더를 통해 이미지 복원
                    reconstructed_latents = reconstructed_latents / vae.config.scaling_factor
                    reconstructed_images = vae.decode(reconstructed_latents).sample
                    
                    # 이미지 후처리 및 저장
                    for idx, reconstructed_image in enumerate(reconstructed_images):
                        # 이미지 범위를 [-1, 1]에서 [0, 1]로 변환
                        reconstructed_image = (reconstructed_image / 2 + 0.5).clamp(0, 1)
                    
                        # 텐서를 CPU로 이동하고 PIL 이미지로 변환
                        reconstructed_image = reconstructed_image.cpu()
                        pil_image = to_pil_image(reconstructed_image)
                    
                        # 캡션을 파일명에 사용하기 위해 처리
                        caption = captions[idx]
                    
                        # 파일명에 사용할 수 없는 문자 제거 및 공백 처리
                        # 여기서는 파일명에 안전한 방식으로 URL 인코딩을 사용합니다.
                        sanitized_caption = urllib.parse.quote(caption, safe='')
                    
                        # 파일명 길이 제한 (예: 100자로 제한)
                        max_filename_length = 100
                        if len(sanitized_caption) > max_filename_length:
                            sanitized_caption = sanitized_caption[:max_filename_length]
                    
                        # 이미지 저장 디렉토리 생성
                        save_dir = f"reconstructed_images/epoch_{epoch+1}"
                        os.makedirs(save_dir, exist_ok=True)

                        # 이미지 파일명 지정 및 저장
                        image_filename = f"{save_dir}/step_{step}_sample_{idx}_{sanitized_caption}.png"
                        pil_image.save(image_filename)

                pbar.set_postfix({'loss': loss.item()})
                with open(csv_file_path, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch+1, step, loss.item()])

        # 각 epoch 이후 모델 체크포인트 저장
        pipe.save_pretrained(f"stable-diffusion-v1-5-finetuned-epoch{epoch+1}")