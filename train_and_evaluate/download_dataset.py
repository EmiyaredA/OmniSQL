import os
import shutil
from huggingface_hub import hf_hub_download

# 设置 Hugging Face 镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置目标目录（您想要移动到的目录）
target_dir = "/mnt/cache/tonghao/slns/dbProject2/OmniSQL/train_and_evaluate"
os.makedirs(target_dir, exist_ok=True)

# 下载数据集，默认保存在 ~/.cache/huggingface/hub
file_path = hf_hub_download(
    repo_id="seeklhy/OmniSQL-datasets",
    filename="data.zip",
    repo_type="dataset"
)

# 构造目标路径
target_path = os.path.join(target_dir, os.path.basename(file_path))

# 剪贴文件（从缓存目录移动到目标目录）
shutil.move(file_path, target_path)

print("Moved to:", target_path)
