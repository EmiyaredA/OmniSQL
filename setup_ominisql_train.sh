conda create -n omnisql_train python=3.9.5
conda activate omnisql_train
pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 transformers==4.45.1 accelerate==0.34.2 deepspeed==0.10.3 numpy==1.24.3 peft datasets tensorboard ijson

pip3 install flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

