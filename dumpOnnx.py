
import crnn.crnn as ocr

import torch.onnx
import torchvision
from PIL import Image

def GenInput(imageFileName):
    image = Image.open(imageFileName).convert('L')
    scale = image.size[1]*1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    print(w)

    transformer = ocr.resizeNormalize((w, 32))
    image = transformer(image).cpu()
            
    image = image.view(1, *image.size())
    return image

imageFileName ="./1.jpg"
dummy_input = GenInput(imageFileName)
torch.onnx.export(ocr.model, dummy_input, "models/crnn.onnx", verbose=True)


