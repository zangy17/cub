
import json

with open('class2concepts.json','r') as fp:
    cub2 = json.load(fp)
all_att_orig = []
concept2class ={}
itt=0
class_names = list(cub2.keys())
for id,c in enumerate(cub2):



    for con in cub2[c]:
        all_att_orig.append(con)
        concept2class[itt]=[id]
        itt+=1
    '''
    for con in cub3[c]:
        all_att_orig.append(con)
        concept2class[itt]=[id]
        itt+=1
    '''

all_att_orig = list(set(all_att_orig))
print(len(all_att_orig))
import open_clip
import torch
from tqdm import tqdm
model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai',cache_dir='../clip')
tokenizer = open_clip.get_tokenizer('ViT-L-14')
device = 'cuda:0'
model = model.to(device)


all_embeddings = []
new_concepts = []
matches = []
new_cub={}
for c in class_names:
    new_cub[c] = []
with torch.no_grad():
    for t in tqdm(all_att_orig):
        t = tokenizer(t).to(device)
        t_embed = model.encode_text(t)
        all_embeddings.append(t_embed)

all_embeddings = torch.stack(all_embeddings,dim=1)
all_embeddings = all_embeddings.squeeze(0)
all_embeddings /= all_embeddings.norm(dim=-1,keepdim=True)
sims = 100 * all_embeddings@all_embeddings.T
def same(i,j):
    if sims[i][j]>95:
        return True
    return False
dup =[]
for i in tqdm(range(len(all_att_orig))):
    if i in dup:
        continue
    for j in range(len(all_att_orig)):
        if i<j and same(i,j):
            dup.append(j)
            concept2class[i]+=concept2class[j]

for i in tqdm(range(len(all_att_orig))):
    if i in dup:
        continue
    new_concepts.append(all_att_orig[i])
    for cc in concept2class[i]:
        if all_att_orig[i] not in new_cub[class_names[cc]]:
            new_cub[class_names[cc]].append(all_att_orig[i])
print(len(new_concepts))
with open('concepts_xx2.json','w') as fw:
    json.dump(new_concepts,fw)
with open('labo_xx2.json','w') as fw:
    json.dump(new_cub,fw)
import numpy as np
class_label = np.zeros((len(new_cub),len(new_concepts)))
for i in range(len(new_cub)):
    for j in range(len(new_concepts)):
        if new_concepts[j] in new_cub[class_names[i]]:
            class_label[i][j]=1
np.save('class_label_labo_xx2.npy',class_label)




