#coding:utf-8

import crnn.crnn as ocr

imageFileName ="./2.jpg"
preds =ocr.crnnOcr(imageFileName)
print(preds)

