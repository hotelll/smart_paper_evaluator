# -*- coding:utf-8 -*-

import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import PIL.Image as Image
from torchvision import models
from torchvision import transforms
from torch.nn import functional as F
import cv2
import matplotlib as mpl
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('..')
from nn_process import save_jpg

INPUT_DIR = 'D:/Projects/python/Acemap-Paper-X-Ray/input/'
OUTPUT_DIR = 'D:/Projects/python/Acemap-Paper-X-Ray/output/'


def get_nn_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    ckp_path = OUTPUT_DIR + 'nn_output/model_best.pth.tar'

    checkpoint = torch.load(ckp_path)
    d = checkpoint['state_dict']
    d = {k.replace('module.', ''): v for k, v in d.items()}
    model.load_state_dict(d)
    model = model.to(device)
    return model


def load_img(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = Image.open(path)
    trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = trans(img)
    img = img.to(device)
    img = torch.unsqueeze(img, 0)
    return img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, help="pdf_path")
    args = parser.parse_args()
    return args


def draw_heatmap(pdf_path, model, save_path="./heatmap.jpg"):
    tmp_dir = INPUT_DIR + 'temp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    tgt_path = tmp_dir + 'image.jpg'
    save_jpg(pdf_path, tgt_path, tmp_dir)
    img_path = tgt_path
    img = load_img(img_path)
    model.eval()
    features = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )(img)
    pooled = model.avgpool(features).view(1, 512)
    output = model.fc(pooled)

    def extract(g):
        global features_grad
        features_grad = g

    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()
    grads = features_grad
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    pooled_grads = pooled_grads[0]
    features = features[0]

    for j in range(512):
        features[j, ...] *= pooled_grads[j, ...]

    heatmap = features.detach().to(torch.device('cpu')).numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    plt.matshow(heatmap)
    plt.show()

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
    cv2.imwrite(save_path, superimposed_img)

    if os.path.exists(tgt_path):
        os.remove(tgt_path)

    return save_path


def predict(pdf_path, model):
    tmp_dir = INPUT_DIR + 'temp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    tgt_path = tmp_dir + 'image.jpg'
    save_jpg(pdf_path, tgt_path, tmp_dir)
    img = load_img(tgt_path)
    logit = model(img)
    probs = F.softmax(logit.to(torch.device("cpu")) / 5, dim=1).data.squeeze()
    nn_score = probs[0].numpy()
    return round(nn_score * 100, 0)


def main():
    pdf_path = "./Abdulnabi_Episodic_CAMN_Contextual_CVPR_2017_paper.pdf"
    nn_model_overall = get_nn_model()

    score = predict(pdf_path, nn_model_overall)
    CAM_ov = draw_heatmap(pdf_path, nn_model_overall)
    print("Score: ", score)
    return CAM_ov


if __name__ == '__main__':
    args = parse_args()
    print(main())
