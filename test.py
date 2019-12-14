from __future__ import print_function
import os
import cv2
from models import *
import torch
import numpy as np
import time
from config import Config
from torch.nn import DataParallel

def get_test_dataset(path):
    a_path = []
    b_path = []
    with open(path) as f:
        for line in f:
            line = line.strip('\n')
            info = line.split(',')
            a_path.append(info[0])
            b_path.append(info[1])
    return a_path, b_path

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def test(model, a_list, b_list):
    a_features, a_cnt = get_featurs(model, a_list)
    b_features, b_cnt = get_featurs(model, b_list)
    anses = []
    #  print(a_features.shape)
    for a_f, b_f in zip(a_features, b_features):
        ans = np.dot(a_f, b_f)/(np.linalg.norm(a_f)*(np.linalg.norm(b_f)))
        # ans = tanh(ans-0.3)
        anses.append(ans)
    return anses



def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    # print(img_path)    
    # image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def get_val_dataset(path):
    true_list = []
    a_path = []
    b_path = []
    with open(path) as f:
        for line in f:
            line = line.strip('\n')
            info = line.split(',')
            true_list.append(int(info[0]))
            a_path.append(info[1])
            b_path.append(info[2])
    return true_list, a_path, b_path

def val(model, true_list, a_list, b_list):
    a_features, a_cnt = get_featurs(model, a_list)
    b_features, b_cnt = get_featurs(model, b_list)
    anses = []
    #  print(a_features.shape)
    for a_f, b_f in zip(a_features, b_features):
        ans = np.dot(a_f, b_f)/(np.linalg.norm(a_f)*(np.linalg.norm(b_f)))
        anses.append(ans)
    return roc_auc_score(np.array(true_list), np.array(anses))

def get_featurs(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)
            output = output.data.cpu().numpy()
            
            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            
            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)



def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)



if __name__ == '__main__':

    opt = Config()
    model = resnet_face18(opt.use_se)


    model = DataParallel(model)
    # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(opt.test_model_path))
    model.to(torch.device("cuda"))
    a_path, b_path = get_test_dataset(opt.test_list)

    # identity_list = get_lfw_list(opt.lfw_test_list)
    # img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    model.eval()
    anses = test(model, a_path, b_path)
    with open("submission.txt", "w") as f:
        for line in anses:
            f.write(f'{abs(line)}\n')
    # lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
