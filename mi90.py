import sklearn
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency
'''
x1 = [0,0,0,1,0,1,0,0,0]
x2 = [0,0,0,0,0,0,0,0,1]
y = [0.1,0.2,0.1,0.8,0.75,0.62,0.33,0.19,0.17]
#print(mutual_info_regression([[t] for t in x1],y))
#print(mutual_info_score(x1,y))

'''
from tqdm import tqdm
import os
import numpy as np
import torch
import open_clip
from PIL import Image
from torch import nn
import pandas as pd
import json
def load_data(data_root, split):
    # Load image data
    images = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "images.txt"),
        sep=" ", names=["image_id", "filepath"],
    )
    image_class_labels = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "image_class_labels.txt"),
        sep=" ", names=["image_id", "class_id"],
    )
    train_test_split = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "train_test_split.txt"),
        sep=" ", names=["image_id", "is_training_image"],
    )
    classes = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "classes.txt"),
        sep=" ", names=["class_id", "class_name"],
    )

    data = images.merge(image_class_labels, on="image_id")
    data = data.merge(train_test_split, on="image_id")
    data = data.merge(classes, on="class_id")

    # Get data split
    if split == "train":
        data = data[data.is_training_image == 1]
    elif split == "valid":
        data = data[data.is_training_image == 0]
    elif split == "all":
        data = data

    data["class_name"] = [class_name.split(".")[1].lower().replace("_", " ") for class_name in data.class_name]
    data['label'] = np.array(data["class_id"]).astype(np.int)

    # Load attribute data
    image_attribute_labels = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "attributes", "image_attribute_labels.txt"),
        sep=" ", names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
    )
    attributes = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "attributes", "attributes.txt"),
        sep=" ", names=["attribute_id", "attribute_name"]
    )
    attributes_info = [attr.split("::") for attr in attributes.attribute_name]
    attributes_info = np.array([[attr.replace("_", " "), label.replace("_", " ")] for attr, label in attributes_info])
    attributes["attribute_template"] = attributes_info[:, 0]
    attributes["attribute_label"] = attributes_info[:, 1]
    attributes = image_attribute_labels.merge(attributes, on="attribute_id")
    unique_attributes = attributes.attribute_template.unique()
    return data, attributes, unique_attributes
model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai',cache_dir='../clip')
tokenizer = open_clip.get_tokenizer('ViT-L-14')
#model.load_state_dict(torch.load('checkpoint/cub_des_xo99.pt', map_location='cuda:0'))


device = 'cuda:0'
model.to(device)
with torch.no_grad():
    model.eval()
    data, attributes, unique_attributes = load_data('/mnt/localssd', split="train")
    image_embeddings = []
    labels = []
    label_dict = {}
    for row_id in tqdm(range(len(data))):
        # Prepare image inputs
        image_id, label, image_name, is_training = data.iloc[row_id][
            ["image_id", "label", "filepath", "is_training_image"]]
        labels.append(label-1)
        if label-1 not in label_dict:
            label_dict[label-1]=[]
        label_dict[label-1].append(row_id)
    if os.path.exists('image_emd_train.pt'):
        image_embeddings = torch.load('image_emd_train.pt')
    else:
        for row_id in tqdm(range(len(data))):
            # Prepare image inputs
            image_id, label, image_name, is_training = data.iloc[row_id][
                ["image_id", "label", "filepath", "is_training_image"]]
            image_path = os.path.join("/mnt/localssd", "CUB_200_2011", "images", image_name)
            image_input = test_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            image_emd = model.encode_image(image_input)
            image_embeddings.append(image_emd)
        image_embeddings = torch.stack(image_embeddings,dim=1)
        image_embeddings = image_embeddings.squeeze(0)
        image_embeddings /= image_embeddings.norm(dim=-1,keepdim=True)
        torch.save(image_embeddings,'image_emd_train.pt')
    with open('all_att_90.json','r') as fp:
        concepts=json.load(fp)
    class_label=np.load('class_label_des_90.npy')
    mutual_info = []
    for i in tqdm(range(len(concepts))):
        x = []
        cand_ids = np.random.choice(len(data),len(data),replace=False)
        pos_ct=0
        neg_ct=0
        choose_id = []
        for c_id in cand_ids:
            if class_label[labels[c_id]][i]==1:
                if pos_ct>=30:
                    continue
                choose_id.append(c_id)
                x.append([1])
                pos_ct+=1
        for c_id in cand_ids:
            if class_label[labels[c_id]][i] == 0:
                if neg_ct >= pos_ct:
                    continue
                choose_id.append(c_id)
                x.append([0])
                neg_ct += 1

        #print(len(choose_id))
        choose_id = np.array(choose_id)
        text = tokenizer(concepts[i]).to(device)
        t_embed = model.encode_text(text)
        t_embed /= t_embed.norm(dim=-1,keepdim=True)
        y = t_embed@image_embeddings[choose_id].T
        y = y.squeeze(0)
        #y = torch.nn.functional.softmax(y,dim=0)
        y = y.cpu().numpy()
        if i==12:
            print([t[0] for t in x])
            print(list(y))
        mutual_info.append(mutual_info_regression(x, y)[0])
#print(mutual_info)
ids = np.argsort(mutual_info)[::-1]
mutual_info = np.array(mutual_info)
print(0,mutual_info[ids[0]])
print(100,mutual_info[ids[100]])
print(500,mutual_info[ids[500]])
print(1000,mutual_info[ids[1000]])
print(1500,mutual_info[ids[1500]])
#print(2000,mutual_info[ids[2000]])
np.save('mutual_info90.npy',mutual_info)


