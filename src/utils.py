import torch

from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import cv2
import glob

from torchvision import transforms as T
import torchvision.transforms as transforms

def to_var(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    """Convert variable to tensor."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data

# custom weights initialization called on networks
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != "ConvEncoder":
        m.weight.data.normal_(0.0, 0.02)
        if not(m.bias is None):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# view images
def visualize(img_arr):
    plt.imshow(((img_arr.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')

# load one image in tensor format
def load_image(filename, load_type=0, wd=256, ht=256):
    #centerCrop = transforms.CenterCrop((wd, ht))
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    if load_type == 0:
        img = transform(Image.open(filename).convert('RGB'))
    else:
        img = transform(text_image_preprocessing(filename))
        
    return img.unsqueeze(dim=0)

def save_image(img, filename):
    tmp = ((img.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))

def save_semantic_image(img, filename):
    img = img.argmax(dim=0).unsqueeze(0).numpy().transpose(1, 2, 0).astype(np.uint8)
    cv2.imwrite(filename, img)

# binarize an image
def binarize(input, threshold=0):
    mask = input > threshold
    input[mask] = 1
    input[~mask] = -1
    return input

def pil2cv(image):
    ''' PIL -> OpenCV '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    ''' OpenCV -> PIL '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def get_img_paths(dir, num=None):
        img_paths = glob.glob(dir + '/**')
        if num != None:
            img_paths = [img_paths[n] for n in range(num)]

        img_paths.sort()

        return img_paths

def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram