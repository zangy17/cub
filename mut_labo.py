import numpy as np
import json
mutual_info = np.load('mutual_info_labo.npy')

ids = np.argsort(mutual_info)[::-1]

#print(1500,mutual_info[ids[1500]])
#print(2000,mutual_info[ids[2000]])

#print(2000,mutual_info_s[ids2[2000]])

with open('concepts.json', 'r') as fp:
    concepts = json.load(fp)
class_label=np.load('class_label_labo_new.npy')
class_label = class_label.T

for i in range(10,20):
    print(ids[-i],concepts[ids[-i]])
    t = []
    for j in range(200):
        if class_label[ids[-i]][j]==1:
            t.append(j)
    print(t)

for i in range(len(ids)):
    if ids[i]==12:
        print(i)
class_ct = {}
class_concept_id = {}
for c in range(200):
    class_ct[c]=0
    class_concept_id[c]=[]
new_concepts_id=[]
'''
for i in range(len(ids)):
    c_id = ids[i]
    for c in range(200):
        if class_label[c_id][c]==1:
            if class_ct[c]>=5:
                continue
            else:
                class_ct[c] += 1
                if c_id in new_concepts_id:
                    continue
                new_concepts_id.append(c_id)
                for c in range(200):
                    if class_label[c_id][c] == 1:
                        class_concept_id[c].append(c_id)
'''
for i in range(len(class_label)):
    c_id = ids[i]
    flag = 0
    for c in class_ct:
        if class_ct[c] < 2:
            flag = 1
    if flag == 0:
        print('f',i)
        break


    for c in range(200):
        if class_label[c_id][c]==1:
            if class_ct[c]>=2:
                continue
            else:
                class_ct[c] += 1
                if c_id in new_concepts_id:
                    continue
                new_concepts_id.append(c_id)
                for c in range(200):
                    if class_label[c_id][c] == 1:
                        class_concept_id[c].append(c_id)

for i in range(2000):
    c_id = ids[i]
    if len(new_concepts_id)>=400:
        print('f', i)
        break

    for c in range(200):
        if class_label[c_id][c]==1:

            class_ct[c] += 1
            if c_id in new_concepts_id:
                continue
            new_concepts_id.append(c_id)
            for c in range(200):
                if class_label[c_id][c] == 1:
                    class_concept_id[c].append(c_id)
            break
#print(class_ct)
print(class_concept_id)
print(len(ids))
print(len(new_concepts_id))
print(len(set(new_concepts_id)))



new_concepts=[]
new_class_label=class_label[new_concepts_id]
for i in new_concepts_id:
    new_concepts.append(concepts[i])

new_class_label = new_class_label.T
with open('concepts_of_400.json','w') as fw:
    json.dump(new_concepts,fw)
np.save('class_label_labo_of_400.npy',new_class_label)


with open('new_cub_all.json','r') as fp:
    new_cub = json.load(fp)
cub_fff = {}
full_cub_fff = {}
class_names = list(new_cub.keys())
for i in range(len(class_names)):
    n = class_names[i]
    cub_fff[n] = []
    full_cub_fff[n] = []

    for c_id in class_concept_id[i]:
        c = concepts[c_id]
        cub_fff[n].append(c)
        full_cub_fff[n].append('A photo of a '+ n+', which has '+c)

with open('labo_of_400.json','w') as fw:
    json.dump(cub_fff,fw)
with open('full_labo_of_400.json','w') as fw:
    json.dump(full_cub_fff,fw)
tl = new_class_label.T
ct=0
for i in range(len(new_concepts)):
    if np.sum(tl[i])>1:
        ct+=1
print(ct/len(new_concepts))
