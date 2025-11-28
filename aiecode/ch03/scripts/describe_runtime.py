from __future__ import annotations
import torch

def describe_runtime() -> dict[str, str]:
    cuda_ready = torch.cuda.is_available()
    mps_ready = torch.backends.mps.is_available()
    device = "cuda" if cuda_ready else ("mps" if mps_ready else "cpu")
    version = torch.__version__
    name = torch.cuda.get_device_name(0) if device =="cuda" else device.upper()
    return {"device": device, "name": name, "version": version}

if __name__ == "__main__":
    info = describe_runtime()
    print(f"device={info['device']}, name={info['name']}, version={info['version']}")