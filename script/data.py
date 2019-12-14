# %%
import os

# %%
root_path = "/home/lihebeizi/data/FaceRegDataset/train_enhanced"
dataset_dict = {}
for root, dirs, files in os.walk(root_path):
    parent_dirname = os.path.basename(root)
    if parent_dirname not in dataset_dict:
        dataset_dict[parent_dirname] = []
    for file in files:
        file_path = os.path.join(root, file)
        dataset_dict[parent_dirname].append(file_path)

#%%
plain_list = []
with open(f"{root_path}/test.meta", "w") as f:
    for i in range(1, 1001):
        dirname = str(i).zfill(4)
        f.write(f"{dataset_dict[dirname][0]},{dataset_dict[dirname][1]}\n")
    
#%%
plain_list = []
import random
for name in dataset_dict:
    for path in dataset_dict[name]:
        plain_list.append((name, path))
random.shuffle(plain_list)

#%%
cnt = 0
val_list = []
selected_list = []
for i in range(0, 3000, 2):
    first = plain_list[i]
    second = plain_list[i+1]
    if len(val_list) >= 500:
        break
    cnt += 2
    if first[0] == second[0]:
        pass
    else:
        val_list.append((0, first[1], second[1]))
        selected_list.append(first[1])
        selected_list.append(second[1])
# %%
for name in dataset_dict:
    if len(val_list) >= 1000:
        break
    if len(dataset_dict[name]) <= 2:
        continue
    first = dataset_dict[name][0]
    second = dataset_dict[name][1]
    val_list.append((1, first, second))
    selected_list.append(first)
    selected_list.append(second)
# %%
id_cnt = 0
with open(f"{root_path}/train.meta", "w") as f:
    for name in dataset_dict:
        for item in dataset_dict[name]:
            if item in selected_list:
                continue
            f.write(f"{item},{id_cnt}\n")
        id_cnt += 1


# %%
with open(f"{root_path}/val.meta", "w") as f:
    for item in val_list:
        f.write(f"{item[0]},{item[1]},{item[2]}\n")

# %%
