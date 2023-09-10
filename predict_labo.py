import numpy as np
import torch
import clip

from PIL import Image

import json
from tqdm import tqdm
import torch.nn as nn
import time
import os
import open_clip

from cub_data import load_data
from torch.utils.data import Dataset, DataLoader
from imagenet_prompts.standard_image_prompts import cub_templates

PATH_TO_PROMPTS = "full_labo.json"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai',cache_dir='../clip')

tokenizer = open_clip.get_tokenizer('ViT-L-14')


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
data_root = '/mnt/localssd'
data, _, _ = load_data(data_root, split="all")
class_id_dict = {}
with open(PATH_TO_PROMPTS) as f:
    gpt3_prompts = json.load(f)
i = 0
for c in gpt3_prompts:
    class_id_dict[c] = i
    i += 1
class_names = list(class_id_dict.keys())


class CubDataset(Dataset):
    def __init__(self, paths,labels,transform):
        self.paths=paths
        self.labels=labels
        self.transform  =transform
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img, label = Image.open(self.paths[i]), int(self.labels[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, label, i

train_image_paths = []
train_labels = []
test_image_paths = []
test_labels = []
for row_id in tqdm(range(len(data))):
    image_id, image_name, is_training, class_name = data.iloc[row_id][
        ["image_id", "filepath", "is_training_image", "class_name"]]
    image_path = os.path.join(data_root, "CUB_200_2011", "images", image_name)
    label = class_id_dict[class_name]
    if is_training:
        train_image_paths.append(image_path)
        train_labels.append(label)
    else:
        test_image_paths.append(image_path)
        test_labels.append(label)
train_set = CubDataset(train_image_paths,train_labels,test_preprocess)
test_set = CubDataset(test_image_paths,test_labels,test_preprocess)

loader = DataLoader(train_set, batch_size=512, num_workers=8,shuffle=False)

def zeroshot_classifier(textnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        i = 0
        for classname in tqdm(textnames):
            texts = [template.format(textnames[i]) for template in templates]  # format with class
            texts = clip.tokenize(texts).to(device)  # tokenize
            class_embeddings = model_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            i += 1
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights
def zeroshot_classifier_gpt(textnames):
    with torch.no_grad():
        zeroshot_weights = []
        i = 0
        for classname in tqdm(textnames):

            texts = []

            for t in gpt3_prompts[textnames[i]]:
                texts.append(t)
            texts = clip.tokenize(texts, truncate=True).to(device)  # tokenize
            class_embeddings = model_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            i += 1

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights



start = time.time()

    #zeroshot_weights_cupl = zeroshot_classifier_gpt(class_names)

#model.load_state_dict(torch.load('checkpoint/cub_des_text'+str(itt*20+1)+'.pt', map_location='cuda:0'))
model = model.to(device)
zeroshot_weights = zeroshot_classifier_gpt(class_names)



total = 0.
correct_base = 0.
correct_cupl = 0.
correct_both = 0.


start = time.time()
pred_labels = []
with torch.no_grad():
    for i, (images, target, num) in enumerate(tqdm(loader)):
        images = images.to(device)
        target = target.to(device)

        # predict
        image_features = model_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits_cupl = image_features @ zeroshot_weights
        pred_cupl = torch.argmax(logits_cupl, dim=1)
        pred_cupl = list(pred_cupl.cpu().numpy())
        pred_labels+=pred_cupl


for i in range(len(pred_labels)):
    if pred_labels[i]==train_labels[i]:
        correct_cupl+=1
print('acc:',correct_cupl/len(train_labels)*100)
end = time.time()
ct=0
assert len(train_labels)==len(pred_labels)
for i in range(len(train_labels)):
    if train_labels[i]==pred_labels[i]:
        ct+=1
print(ct/len(pred_labels))
import json
from numpyencoder import NumpyEncoder
with open('pred_labels_labo.json','w') as fw:
    json.dump(pred_labels,fw,cls=NumpyEncoder)


