
import torch

data = torch.load("ref_lib/embed_dict.pt")
print("Type:", type(data))

if isinstance(data, dict):
    print("Number of keys:", len(data))
    # for k in data:
    #     print(f"{k}: {type(data[k])}")