import torch

def check_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM (free/total): {torch.cuda.mem_get_info()}")
        
        x = torch.rand(5, 3)
        print("CPU Tensor:")
        print(x)
        
        x = x.cuda()
        print("GPU Tensor:")
        print(x)
        
        x = x.cpu()

if __name__ == "__main__":
    check_cuda()

