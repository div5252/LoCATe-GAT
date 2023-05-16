from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from modules.visual import transformer_network
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import cv2
import numpy as np
import argparse
import os
from dataset import dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ucf', type=str, help='Dataset: [ucf, hmdb, olympics, test]')
parser.add_argument('--dataset_path', required=True, type=str, help='Path of the datasets')
parser.add_argument('--ckp_path', type=str, help='Path of checkpoint model')
parser.add_argument('--layer', type=str, required=True, help='Layer: [ln1, ln2, lca]')
parser.add_argument('--save_path', required=True, type=str, help='Where to save visualization result.')
opt = parser.parse_args()

def reshape_transform_ln(tensor):
    # LN 1 & LN 2
    tensor = torch.mean(tensor, 1)
    result = tensor.reshape(tensor.size(0), 2, 16, 16)
    return result

def reshape_transform_lca(tensor):
    # LCA branches
    tensor = torch.mean(tensor, 0)
    tensor = tensor.reshape(1, tensor.size(0), tensor.size(1), tensor.size(2))
    return tensor

def gradcam_output(fnames, ucf_class, model, opt):
    if opt.layer == 'ln1':
        target_layers = [model.transformer_block1.ln_1]
        reshape_transform = reshape_transform_ln
    elif opt.layer == 'ln2':
        target_layers = [model.transformer_block1.ln_2]
        reshape_transform = reshape_transform_ln
    elif opt.layer == 'lca':
        target_layers = [model.transformer_block1.lca.branch1[5], model.transformer_block1.lca.branch2[5], model.transformer_block1.lca.branch3[5]]
        reshape_transform = reshape_transform_lca

    cam = GradCAM(model=model,
                target_layers=target_layers,
                use_cuda=True,
                reshape_transform=reshape_transform)

    input_tensor = []
    for fname in fnames:
        rgb_img = cv2.imread(os.path.join(opt.save_path, 'frame', ucf_class, fname), 1)
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255

        tensor = preprocess_image(rgb_img, mean=[0.43216, 0.394666, 0.37645],
                                        std=[0.22803, 0.22145, 0.216989])
        input_tensor.append(tensor)
    
    input_tensor = torch.cat(input_tensor, dim=0)
    l, ch, h, w = input_tensor.shape
    input_tensor = input_tensor.reshape(1, 1, ch, l, h, w)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    return cam_image


if __name__ == '__main__':
    ucf_fnames, ucf_labels, ucf_classes, folder = dataset.get_test_data(opt.dataset, opt.dataset_path)

    model = transformer_network.CLIPTransformer(16, [0,0,0])
    # load weights
    j = len('module.')
    weights = torch.load(opt.ckp_path)['state_dict']
    model_dict = model.state_dict()
    weights = {k[j:]: v for k, v in weights.items() if k[j:] in model_dict.keys()}
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    model.eval()
    model = model.cuda()

    os.mkdir(os.path.join(opt.save_path, opt.layer))
    for ucf_class in ucf_classes:
        os.mkdir(os.path.join(opt.save_path, opt.layer, ucf_class))

    for ucf_class in tqdm(ucf_classes):
        files = sorted(os.listdir(os.path.join(opt.save_path, 'frame', ucf_class)))
        for i in range(2):
            cam_image = gradcam_output(files[i*16:(i+1)*16], ucf_class, model, opt)
            output_path = os.path.join(opt.save_path, opt.layer, ucf_class, files[i*16])
            cv2.imwrite(output_path, cam_image)