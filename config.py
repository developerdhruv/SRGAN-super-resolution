import torch
from PIL import Image
import albumentatios as A
from albumentation.pytorch import ToTensorV2

LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
NUM_EPOCHS = 100000
HIGH_RES = 256
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3
highres_transform = A.Compose(
    [
    A.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
    ToTensorV2()
    ]
)


lowres_transform = A.Compose(
    [
    A.Resize(height=LOW_RES, width=LOW_RES, interpolation=Image.BICUBIC),
    A.Normalize(mean = [0, 0, 0], std = [1, 1, 1]),
    ToTensorV2()
    ]
)


both_transform = A.Compose(
    [
    A.Resize(height=HIGH_RES, width=HIGH_RES, ),
    A.HorizontalFlip(p=0.5),
    A.randomRotate90(p=0.5),
    ]
)