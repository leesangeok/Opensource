import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from diffusers import StableDiffusion3Pipeline
from transformers import AdamW
from accelerate import Accelerator
from tqdm import tqdm
from datetime import datetime
import time
import subprocess
import csv

# GPU 메모리 사용 설정
def is_gpu_available(min_free_memory=20000):
    """
    GPU가 사용 가능한지 확인하는 함수.
    min_free_memory: 사용 가능한 메모리의 최소 용량 (MiB 단위)
    """
    try:
        # nvidia-smi 명령어 실행
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE,
            text=True
        )
        # 각 GPU의 사용 가능한 메모리를 리스트로 저장
        free_memory = [int(x) for x in result.stdout.strip().split("\n")]
        # 사용 가능한 메모리가 설정한 용량보다 큰 GPU가 있으면 True 반환
        return any(memory > min_free_memory for memory in free_memory)
    except Exception as e:
        print(f"Error checking GPU availability: {e}")
        return False

# GPU가 사용 가능해질 때까지 대기
print("Waiting for a free GPU...")
while not is_gpu_available():
    print("Waiting.." + datetime.now().strftime("%Y%m%d_%H%M%S"))
    time.sleep(10)  # 10초 대기 후 다시 확인

print("GPU is now available. Starting training...")

# CSV 파일 초기화
csv_file_path = "training_loss_log.csv"
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss"])  # 헤더 작성

# torch.cuda.set_per_process_memory_fraction(0.7, device=torch.cuda.current_device())
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 저장된 모델 경로
save_path = "/home/oss_1/MinsuKim/stable-diffusion-3.5-model"
# 모델 불러오기
pipe = StableDiffusion3Pipeline.from_pretrained(save_path, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

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
                if image_file.lower().endswith(('.png')):
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

# 이미지 전처리 설정 (1024x1024 크기로 조정, 정규화)
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 데이터셋 및 DataLoader 생성
train_image_root = "/home/oss_1/data/240.심볼(로고) 생성 데이터/01-1.정식개방데이터/Training/01.원천데이터"
train_label_root = "/home/oss_1/data/240.심볼(로고) 생성 데이터/01-1.정식개방데이터/Training/02.라벨링데이터"

train_dataset = CustomDataset(train_image_root, train_label_root, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers= 8, pin_memory=True)


text_encoder = pipe.text_encoder
transformer = pipe.transformer

vae = pipe.vae.to("cuda")
scheduler = pipe.scheduler
# 옵티마이저 및 Accelerator 설정
optimizer = AdamW(
    list(pipe.text_encoder.parameters()) +
    list(pipe.transformer.parameters()), 
    lr=1e-5
)

# 분산 학습과 혼합 정밀도 학습을 간소화
accelerator = Accelerator()
text_encoder, transformer, optimizer = accelerator.prepare(text_encoder, transformer, optimizer)
train_dataloader = accelerator.prepare(train_dataloader)

criterion = torch.nn.MSELoss()
# 학습 루프
num_epochs = 3
for epoch in range(num_epochs):
    # 학습 모드 설정
    transformer.train()
    text_encoder.train()

    for images, captions in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
        images = images.to(accelerator.device, dtype=torch.bfloat16)
        print(captions)
        inputs = pipe.tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(accelerator.device)
        print(input_ids.shape)
        attention_mask = inputs["attention_mask"].to(accelerator.device)

        # 2. 텍스트 임베딩 생성
        text_embeddings = pipe.text_encoder(input_ids, attention_mask = attention_mask).last_hidden_state  # 텍스트 인코딩을 통해 텍스트 임베딩 생성
        print("text_embeddings shape:", text_embeddings.shape)  # 디버깅용
        
        context_projection_layer = torch.nn.Linear(text_embeddings.size(-1), 1536).to(accelerator.device, dtype=torch.bfloat16)
        encoder_hidden_states = context_projection_layer(text_embeddings)
        print("encoder_hidden_states shape:", encoder_hidden_states.shape)  # 디버깅용


        # VAE로 이미지를 잠재 공간으로 인코딩
        latent_vectors = vae.encode(images).latent_dist.sample() * 0.18215

        # 노이즈 생성
        noise = torch.randn_like(latent_vectors)
        # 노이즈 추가
        noised_latent_vectors = latent_vectors + noise

        # 학습 루프 내에서 스케줄러를 사용해 노이즈 제거 단계 수행
        for t in scheduler.timesteps:

            timestep = torch.tensor([t], dtype=torch.float32).to(accelerator.device)
            # 선형 변환 레이어를 정의하고 bfloat16 타입으로 변환
            # projection_layer = torch.nn.Linear(text_embeddings.size(-1), 2048).to(accelerator.device, dtype=torch.bfloat16)
            # pooled_projections = projection_layer(text_embeddings.mean(dim=1))
            
            # print("pooled_projections shape:", pooled_projections.shape)
            # Transformer를 사용해 노이즈 제거 예측
            outputs = transformer(
                        hidden_states=noised_latent_vectors,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timestep,
                        # pooled_projections=pooled_projections  # 추가된 인자
                    )
            
            # 스케줄러를 통해 노이즈 제거 단계 진행
            noised_latent_vectors = scheduler.step(outputs, t, noised_latent_vectors).prev_sample

        
       
        # 4. 손실 계산
        loss = criterion(outputs, noise)  # 모델의 출력과 실제 노이즈 간의 손실 계산
        
        # 5. 역전파 및 옵티마이저 업데이트
        optimizer.zero_grad()  # 이전의 그래디언트를 초기화
        accelerator.backward(loss)  # 손실 함수에 대해 역전파 수행
        optimizer.step()  # 옵티마이저를 사용해 모델의 파라미터를 업데이트

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
       # 에포크마다 손실 값을 CSV에 기록
    with open(csv_file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, loss.item()])

print("Complete!")

# 학습 종료 후 모델 저장
save_path = save_path + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
pipe.save_pretrained(save_path)