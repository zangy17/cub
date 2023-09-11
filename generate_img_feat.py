import argparse
import json
import os
import logging

import clip
import pandas as pd
import numpy as np
import torch
import open_clip
from PIL import Image

from cub_data import load_data
from tqdm import tqdm
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"






def main():
    # Load data

    data, attributes, unique_attributes = load_data('/mnt/localssd/', split="train")

    # Load the model

    model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai',cache_dir='../clip')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    #model.load_state_dict(torch.load('checkpoint/cub_des_all49.pt', map_location='cuda:0'))
    class ImageCLIP(nn.Module):
        def __init__(self, model):
            super(ImageCLIP, self).__init__()
            self.model = model

        def forward(self, image):
            return self.model.encode_image(image)

    class TextCLIP(nn.Module):
        def __init__(self, model):
            super(TextCLIP, self).__init__()
            self.model = model

        def forward(self, text):
            return self.model.encode_text(text)

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model_text = nn.DataParallel(model_text)
        model_image = nn.DataParallel(model_image)
    model_text.to(device)
    model_image.to(device)
    model_text.eval()
    model_image.eval()




    labels = []
    image_embeds = []
    for row_id in tqdm(range(len(data))):
        # Prepare image inputs
        image_id, class_id, image_name, is_training = data.iloc[row_id][
            ["image_id", "class_id", "filepath", "is_training_image"]]
        labels.append(int(class_id)-1)
        image_path = os.path.join("/mnt/localssd","CUB_200_2011", "images", image_name)
        image_input = test_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_attributes = attributes[attributes.image_id == image_id]

        with torch.no_grad():
            image_features = model_image(image_input)

        # Pick k most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_embeds.append(image_features)
    torch.save(image_embeds,'image_feat.pt')
    num_image_class=[]
    for i in range(200):
        ct = 0
        for l in labels:
            if l==i:
                ct+=1
        num_image_class.append(ct)
    np.save('num_image_class.npy',np.array(num_image_class))
    print(labels)



if __name__ == "__main__":


    main()
