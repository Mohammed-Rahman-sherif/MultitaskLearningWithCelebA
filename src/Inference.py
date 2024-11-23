# %%
import torch
import requests
import Architecture
from PIL import Image
from io import BytesIO
from torchvision.transforms import Normalize, Compose, ToTensor

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Accelerator: {device}")

# %%
transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# %%
with open("data/celeba/list_attr_celeba.txt", "r") as F:
    lines = F.readlines()
    keys = lines[1].split()
    keys = keys[:40]
print(keys)


# %%
new_model = Architecture.MTArchitecture().to(device=device)
new_model.load_state_dict(torch.load(r"models\best_CelebA_0.6479514565891943.pt", map_location=device))
print(new_model)

# %%
def Inference_image(Image_path, local):
    if not local:
        response = requests.get(Image_path)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(Image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    new_model.eval()
    with torch.no_grad():
        predictions = new_model(image)
        # predictions = [torch.sigmoid(output).item() for output in outputs]

    binary_predictions = [1 if prediction >
                          0.5 else 0 for prediction in predictions]

    for attr, value in zip(keys, binary_predictions):
        print(f'''{attr}: {"yes" if value == 1 else "No"}''')


# %%
IMAGE = r"C:\Users\smart\Documents\Computer Vision\MultiTask Learning\CelebA\data\celeba\img_align_celeba\000125.jpg"
img = Inference_image(IMAGE, local=True)
# %%
