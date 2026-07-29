import torch

# 创建数据
x = torch.tensor([1.0, 2.0])

# 如果有苹果GPU，就放到GPU上
if torch.backends.mps.is_available():
    x = x.to("mps")
    print("✅ PyTorch 使用苹果GPU加速")
else:
    print("⚠️ 仅使用CPU")