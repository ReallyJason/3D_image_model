import sys
import traceback

try:
    sys.path.insert(0, "models/triposr")
    import torch
    from PIL import Image
    from tsr.system import TSR

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading TripoSR on {device}...")
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    print("from_pretrained finished!")
    model.to(device)
    print("model.to(device) finished!")
except Exception as e:
    traceback.print_exc()
