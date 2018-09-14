#coding:utf-8

import torch
import torchvision.transforms as transforms
from net import crnnnet as net
from crnn import keys
from . import util

from collections import OrderedDict
from PIL import Image


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


def crnnSource():
    alphabet = keys.alphabet
    converter = util.strLabelConverter(alphabet)

    model = net.CRNNNET(32, 1, len(alphabet)+1, 256, 1).cpu()

    state_dict = torch.load("./models/ocr.pth", map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.','') # remove `module.`
        new_state_dict[name] = v
    # load params
   
    model.load_state_dict(new_state_dict)
    model.eval()
    
    return model,converter

##加载模型
model,converter = crnnSource()

def crnnOcr(imageFileName):
    """
    crnn模型，ocr识别
    @@model,
    @@converter,
    @@im
    @@text_recs:text box
    """
    image = Image.open(imageFileName).convert('L')
    scale = image.size[1]*1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    print(w)

    transformer = resizeNormalize((w, 32))
    image = transformer(image).cpu()
            
    image = image.view(1, *image.size())
    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = torch.IntTensor([preds.size(0)])
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return sim_pred
       

