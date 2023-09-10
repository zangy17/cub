import os
import numpy as np
import pandas as pd
import torch
import open_clip
import argparse
import pdb

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from torch.utils.data import DataLoader
from PIL import Image
from cub_data import load_data


def get_few_shot_idx(train_y, K, classes):
    '''
    inputs:
        train_y: training labels from 0-199
        K: number of training examples per class (k-shot)
        classes: class labels chosen for task (n-way)

    outputs:
        training indices for n-way k-shot
    '''
    idx = np.arange(6000)
    idx[-6:] -= 6
    idx = idx.reshape((200, -1))
    ignore = [np.random.shuffle(x) for x in idx]

    idx = idx[:, :K].flatten()

    corr = np.concatenate([[i] * K for i in range(200)])
    y = train_y[idx]

    # CORRECT DISTRIBUTION IF NOT UNIFORM
    foo = (y == corr)
    for i, val in enumerate(foo):
        if not val:
            while train_y[idx[i]] < corr[i]:
                idx[i] += 1
            while train_y[idx[i]] > corr[i]:
                idx[i] -= 1

    y = train_y[idx]

    classes_idx = np.argwhere(np.in1d(y, classes)).flatten()
    idx = idx[classes_idx]

    return idx



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

    data["image_id"] = data["image_id"].astype(int)
    data.sort_values(by="image_id")

    data["class_id"] = data["class_id"].astype(int)

    y_labels = np.array(data["class_id"]).astype(np.int)
    y_labels -= 1

    return data, y_labels, unique_attributes, classes
def main():


    # Load the model



    # LOAD TRAIN IMAGES
    train_data, train_y, _, classes = load_data('/mnt/localssd', split="train")

    # LOAD TEST IMAGES
    test_data, test_y, _, classes = load_data('/mnt/localssd', split="valid")
    cls_truth = np.load('class_label_des_f95.npy')


    # primitive concept activations
    tr_att_activations = torch.Tensor(np.load('save_des' + "/attribute_activations_train_95.npy"))
    t_att_activations = torch.Tensor(np.load('save_des' + "/attribute_activations_valid_95.npy"))

    tr_ground_truth = cls_truth[train_y]
    t_ground_truth = cls_truth[test_y]

    print(len(t_ground_truth))
    n = [5,10,20,50,100]
    k = [1,5]

    print('full shot')
    rr = []
    # ConceptCLIP - Primitive (Logistic Regression)
    classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
    classifier.fit(tr_att_activations, train_y)
    lr_score = classifier.score(t_att_activations, test_y)
    rr.append(lr_score)
    # Full - Intervene (Logistic Regression)
    lr_score = classifier.score(t_ground_truth, test_y)
    rr.append(lr_score)
    # Part - Intervene (Logistic Regression)
    prims = t_att_activations
    grounds = t_ground_truth
    part_invs = []
    for i in range(len(prims)):
        part = []
        for j in range(len(prims[i])):
            if grounds[i][j] == 1:
                part.append(1)
            else:
                part.append(prims[i][j])
        part_invs.append(part)
    part_invs = np.array(part_invs)
    lr_score = classifier.score(part_invs, test_y)
    rr.append(lr_score)



    print(rr)


    for N in n:
        for K in k:
            # if N!=200:
            # continue
            results = []
            if K==1:
                t=20
            else:
                t=5
            for exp in tqdm(range(t)):
                curr_result = []

                # GET N CLASS LABELSn
                classes = np.arange(200)
                np.random.shuffle(classes)
                classes = classes[:N]

                # GET K TRAINING IMAGES PER CLASS
                idx = get_few_shot_idx(train_y, K, classes)

                train_y2 = train_y[idx]

                tr_prim = tr_att_activations[idx]

                tr_ground = tr_ground_truth[idx]

                # GET TESTING IMAGES FROM N CLASSES
                t_idx = np.argwhere(np.in1d(test_y, classes)).flatten()

                test_y2 = test_y[t_idx]

                t_prim = t_att_activations[t_idx]

                t_ground = t_ground_truth[t_idx]



                # ConceptCLIP - Primitive (Logistic Regression)
                classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
                classifier.fit(tr_prim, train_y2)
                lr_score = classifier.score(t_prim, test_y2)
                curr_result.append(lr_score)

                results.append(curr_result)
                # Full - Intervene (Logistic Regression)
                lr_score = classifier.score(t_ground, test_y2)
                curr_result.append(lr_score)
                # Part - Intervene (Logistic Regression)
                prims = t_prim
                grounds = t_ground
                part_invs = []
                for i in range(len(prims)):
                    part = []
                    for j in range(len(prims[i])):
                        if grounds[i][j] == 1:
                            part.append(1)
                        else:
                            part.append(prims[i][j])
                    part_invs.append(part)
                part_invs = np.array(part_invs)
                lr_score = classifier.score(part_invs, test_y2)
                curr_result.append(lr_score)


            results = np.array(results)
            print(N, K, np.mean(results, axis=0), flush=True)


if __name__ == "__main__":
    main()

