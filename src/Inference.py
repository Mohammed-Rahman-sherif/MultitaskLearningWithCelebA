# %%
import cv2
import cv2.img_hash
import torch
import requests
import Architecture
from PIL import Image
from io import BytesIO
import numpy as np
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
new_model.load_state_dict(torch.load(
    r"models\best_CelebA_0.6479514565891943.pt", map_location=device))
print(new_model)

# %%


def Inference_image(Image_path, local):
    if not local:
        response = requests.get(Image_path)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(Image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    new_model.eval()
    with torch.no_grad():
        predictions = new_model(image_tensor)
        # predictions = [torch.sigmoid(output).item() for output in outputs]

    binary_predictions = [1 if prediction >
                          0.5 else 0 for prediction in predictions]

    for attr, value in zip(keys, binary_predictions):
        print(f'''{attr}: {"yes" if value == 1 else "No"}''')

    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    # Add predicted attributes as text overlay on the image
    no_offset = 20
    yes_offset = 20
    for attribute, value in zip(keys, binary_predictions):
        text = f"{attribute}: {'Yes' if value == 1 else 'No'}"
        if value == 0:
            cv2.putText(open_cv_image, text, (10, no_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            no_offset += 15
        else:
            cv2.putText(open_cv_image, text, (600, yes_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            yes_offset += 15

    cv2.imwrite("output.png", open_cv_image)

    # Display the image with predictions
    cv2.imshow('Predicted Attributes', open_cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# %%
IMAGE = r"https://img.theweek.in/content/dam/week/magazine/theweek/sports/images/2020/4/23/Sachin-Tendulkar2.jpg"
img = Inference_image(IMAGE, local=False)
# %%
