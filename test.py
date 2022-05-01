import requests
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as transforms
from utils import *
import io
from PIL import Image


resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('sample_image.jpg','rb')})

print(resp.json())
