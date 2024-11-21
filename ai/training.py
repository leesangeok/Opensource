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
    time.sleep(60)  # 60초 대기 후 다시 확인

print("GPU is now available. Starting training...")

torch.cuda.set_per_process_memory_fraction(0.7, device=torch.cuda.current_device())
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
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers= 8, pin_memory=True)

# 텍스트 인코더 가져오기
text_encoder = pipe.text_encoder

# 옵티마이저 및 Accelerator 설정
optimizer = AdamW(text_encoder.parameters(), lr=1e-5)
accelerator = Accelerator()
text_encoder, optimizer = accelerator.prepare(text_encoder, optimizer)

# 학습 루프
num_epochs = 3
for epoch in range(num_epochs):
    for images, captions in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
        images = images.to(accelerator.device)

        # 텍스트 인코딩
        text_inputs = pipe.tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
        text_embeddings = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]

        noise = torch.randn_like(text_embeddings).to(accelerator.device)
        loss = torch.nn.functional.mse_loss(text_embeddings, noise)

        # 역전파 및 옵티마이저 업데이트
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

print("Fine-tuning Complete!")

# 학습 종료 후 모델 저장
save_path = "/home/oss_1/MinsuKim/stable-diffusion-3.5-model_" + datetime.now().strftime("%Y%m%d_%H%M%S")
pipe.save_pretrained(save_path)