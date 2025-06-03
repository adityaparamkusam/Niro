# 1️⃣  create env
conda env create -f env.yml
conda activate niro           # or python -m venv + pip install -r ...

# 2️⃣  one-time tokenizer
python tokenizer_setup.py

# 3️⃣  pre-train
accelerate config   # choose CUDA + fp16 on AWS
accelerate launch train_pretrain.py --batch 64 --epochs 5

# 4️⃣  fine-tune (optional)
accelerate launch train_finetune.py --checkpoint ckpts/pytorch_model.bin

# 5️⃣  prune + quantise
python prune_quantize.py --ckpt ckpts/pytorch_model.bin

# 6️⃣  export ONNX
python export_onnx.py --ckpt ckpts/pytorch_model_pruned_int8.pt

# 7️⃣  chat!
python infer.py --backend torch   # or --backend onnx
