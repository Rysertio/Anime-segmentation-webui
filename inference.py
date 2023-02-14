import os

import argparse
import cv2
import torch
import numpy as np
import glob
from torch.cuda import amp
from tqdm import tqdm
import streamlit as st
from train import AnimeSegmentation
from PIL import Image


def get_mask(model, input_img, use_amp=True, s=640):
    h0, w0 = h, w = input_img.shape[0], input_img.shape[1]
    if h > w:
        h, w = s, int(s * w / h)
    else:
        h, w = int(s * h / w), s
    ph, pw = s - h, s - w
    tmpImg = np.zeros([s, s, 3], dtype=np.float32)
    tmpImg[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h)) / 255
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = torch.from_numpy(tmpImg).unsqueeze(0).type(torch.FloatTensor).to(model.device)
    with torch.no_grad():
        if use_amp:
            with amp.autocast():
                pred = model(tmpImg)
            pred = pred.to(dtype=torch.float32)
        else:
            pred = model(tmpImg)
        pred = pred[0, :, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        pred = cv2.resize(pred.cpu().numpy().transpose((1, 2, 0)), (w0, h0))[:, :, np.newaxis]
        return pred


device = torch.device('cpu')

st.title('Anime Segmentation')

model = AnimeSegmentation.try_load('isnet_is', './isnetis.ckpt', 'cpu')
model.eval()
model.to(device)
st.text('Model loaded')
only_matted = st.radio('Only matted', [True, False])
filea = st.file_uploader('Upload an image')

if (filea is not None):
    img = Image.open(filea)
    img = img.save("img.png")

    # OpenCv Read
    img = cv2.imread("img.png")
    img = cv2.cvtColor(cv2.imread("img.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = get_mask(model, img, use_amp=not False, s=1024)
    if only_matted:
        img = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite('img.png', img)
    else:
        img = np.concatenate((img, mask * img, mask.repeat(3, 2) * 255), axis=1).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('img.png', img)
