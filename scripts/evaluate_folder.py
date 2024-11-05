import argparse
import os
import sys
import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, '.')
from iharm.inference.predictor import Predictor
from iharm.inference.utils import load_model
from iharm.mconfigs import ALL_MCONFIGS

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='', help='Source directory')
parser.add_argument('--des', default='', help='Source directory')
parser.add_argument('--model_type', default='AICT', choices=ALL_MCONFIGS.keys())
parser.add_argument('--weights', default='', help='path to the weights')
parser.add_argument('--gpu', type=str, default='cuda', help='ID of used GPU.')

args = parser.parse_args()

# Load model 
model = load_model(args.model_type, args.weights, verbose=False)

device = torch.device(args.gpu)
use_attn = ALL_MCONFIGS[args.model_type]['params']['use_attn']
normalization = ALL_MCONFIGS[args.model_type]['params']['input_normalization']
predictor = Predictor(model, device, use_attn=use_attn, mean=normalization['mean'], std=normalization['std'])

# Get data
comp_list = sorted(os.listdir(os.path.join(args.src, 'composite_images')))

os.makedirs(args.des, exist_ok=True)

for img in tqdm(comp_list):

    # Load images      
    comp = cv2.imread(os.path.join(args.src, 'composite_images', img))
    comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(args.src, 'masks', img), cv2.IMREAD_GRAYSCALE) / 255
    comp_lr = cv2.resize(comp, (256, 256))
    mask_lr = cv2.resize(mask, (256, 256))
    
    # Inference
    pred_lr, pred_img, color_dis = predictor.predict(comp_lr, comp, mask_lr, mask)

    # Save Image
    min_len = 300
    H, W, _ = pred_img.shape
    L = min(H, W)
    scale = min_len / L
    if scale < 1:
        harm = cv2.resize(pred_img, (int(scale * W), int(scale * H)))

    cv2.imwrite(os.path.join(args.des, img.replace('.png', '') + f'_harm.jpg'), harm[:, :, ::-1])
