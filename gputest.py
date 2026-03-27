import torch

print("🔍 Checking GPU...\n")

# 1. CUDA Availability
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    # 2. GPU Details
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Count:", torch.cuda.device_count())
    
    # 3. Memory Info
    print("\n📊 Memory Info:")
    print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**3, 2), "GB")
    print("Reserved:", round(torch.cuda.memory_reserved(0)/1024**3, 2), "GB")

    # 4. Test Tensor on GPU
    x = torch.rand(3, 3).to("cuda")
    print("\n✅ Tensor successfully created on GPU:")
    print(x)

else:
    print("\n❌ No GPU found. Running on CPU.")