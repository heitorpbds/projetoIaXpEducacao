import torch
if not torch.cuda.is_available():
    print("❌ CUDA não está disponível. Verifique a instalação do driver e do PyTorch.")
else:
    print("✅ CUDA disponível!")
    print(f"GPU Detectada: {torch.cuda.get_device_name(0)}")