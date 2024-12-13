import os
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_root, label_root, transform=None, mode='train'):
        self.image_root = image_root
        self.label_root = label_root
        self.transform = transform
        self.mode = mode
        self.image_paths = []
        self.label_paths = []

        prefix = 'TL_TEXT' if self.mode == 'train' else 'VL_TEXT'

        # 이미지와 레이블 파일 경로를 일치시키는 작업
        for image_folder in os.listdir(image_root):
            if image_folder.endswith('.zip') or image_folder == '.ipynb_checkpoints':
                continue
            image_folder_path = os.path.join(image_root, image_folder)
            label_folder_path = os.path.join(label_root, f"{prefix}_{image_folder.split('_')[1]}_{image_folder.split('_')[2]}")

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
