import numpy as np
import torch
from PIL import Image
import json
from tqdm import tqdm
import torch.nn as nn
import os
import open_clip
import torch.nn.functional as F

from cub_data import load_data
from torch.utils.data import Dataset, DataLoader


PATH_TO_PROMPTS = "new_cub.json"
model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai',cache_dir='../clip')


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)
model_text = TextCLIP(model)
model_image = ImageCLIP(model)

model_text = torch.nn.DataParallel(model_text)
model_image = torch.nn.DataParallel(model_image)
tokenizer = open_clip.get_tokenizer('ViT-L-14')
data, _, _ = load_data(data_root='/mnt/localssd', split="all")
class_id_dict = {}
with open(PATH_TO_PROMPTS) as f:
    gpt3_prompts = json.load(f)
i = 0
for c in gpt3_prompts:
    class_id_dict[c] = i

    i += 1
class_names = list(class_id_dict.keys())
idx_to_text = {}
for c in class_id_dict:
    idx_to_text[class_id_dict[c]]=c

class CubDataset(Dataset):
    def __init__(self, paths,labels,transform,tokenizer):
        self.paths=paths
        self.labels=labels
        self.transform  =transform
        self.tokenize = tokenizer

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img, label = Image.open(self.paths[i]), int(self.labels[i])
        if self.transform is not None:
            img = self.transform(img)
        return img,label,i

train_image_paths = []
train_labels = []
test_image_paths = []
test_labels = []




for row_id in range(len(data)):
    image_id, image_name, is_training, class_name = data.iloc[row_id][
        ["image_id", "filepath", "is_training_image", "class_name"]]
    image_path = os.path.join("/mnt/localssd", "CUB_200_2011", "images", image_name)
    label = class_id_dict[class_name]
    if is_training:
        train_image_paths.append(image_path)
        train_labels.append(label)
    else:
        test_image_paths.append(image_path)
        test_labels.append(label)
import json

with open('pred_labels_of.json','r') as fj:
    train_labels = json.load(fj)
train_set = CubDataset(train_image_paths,train_labels,train_preprocess,tokenizer)
#test_set = CubDataset(test_image_paths,test_labels,test_preprocess,tokenizer,)
print(len(train_set))
#print(len(test_set))


device = 'cuda:0'

model = model.to(device)

world_size = torch.cuda.device_count()
batch_size=512

train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,num_workers=32)

with open('all_att_of.json','r') as fp:
    texts = json.load(fp)
texts = tokenizer(texts)
texts = texts.to(device)



#test(ddp_model)
print("Training ImageNet...")

for (name,param) in model.named_parameters():
    if name!='text_projection' and name!='visual.proj' and name != 'logit_scale':
        param.requires_grad=False
    if name == 'text_projection':
        print(param)
        print(param.shape)
    if name == 'visual.proj':
        print(param)
        print(param.shape)
    if name == 'logit_scale':
        print(param)
params = filter(lambda p:p.requires_grad, model.parameters())
cls_truth = np.load('class_label_des_of.npy')
cls_truth = torch.Tensor(cls_truth).t()
cls_truth = cls_truth.to(device)
#fjirfgurhfh
optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-4)
loss_func = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=100)
for epoch in range(100):

    model.train()
    total = 0.
    correct_cupl = 0.
    avg_loss = 0.

    for (images, targets, num) in tqdm(train_loader):

        images = images.to(device)
        targets = targets.to(device)


        optimizer.zero_grad()
        # predict
        # image_features = model_image(images)
        # image_features = image_features/image_features.norm(dim=-1, keepdim=True)

        # logits_base = image_features @ zeroshot_weights_base
        image_features = model_image(images)
        text_features = model_text(texts)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features,dim=-1)
        pred = torch.matmul(image_features, text_features.t()) * model.logit_scale
        pred = torch.matmul(pred,cls_truth)
        # logits_both = image_features @ zeroshot_weights_gpt_both
        loss = loss_func(pred,targets)
        loss.backward()
        loss_item = loss.detach().cpu().numpy()
        avg_loss += loss_item
        # convert_models_to_fp32(model)
        optimizer.step()
        total += len(images)

    scheduler.step()
    print('epoch:', epoch, 'epoch loss:', np.mean(avg_loss / total * batch_size))
    if epoch %10==9:
        torch.save(model.state_dict(), 'checkpoint/cub_des_of' + str(epoch) + '.pt')
