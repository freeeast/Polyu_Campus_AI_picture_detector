#!/usr/bin/env python
# coding: utf-8

# Target:
# Distinguish between AI-generated and real-life images
# 
# Task steps:
# -Collection of data sets: AI/reality, two cvs, two fold.
# cvs content: id, file name, real tag: AI/real
# 
# -Preprocessing data sets;
# Create a dictionary for all images, create an image_inf class to store cvs data, and process the RGB images to obtain a tensor list and divide the data set train/validation
# 
# -Modeling:
# Mainly focus on the optimization of CNN models, pay attention to the optimization of some feature selection and so on
# 
# -Model output: pred——lable comparison accuracy
# 
# -Model evaluation, use relevant evaluation functions -Or perform backfond optimization, use real photos as testset, and then evaluate the accuracy
# 
# -Visual processing
# 
# -Report
# 
# Bonus points: Identify AI-generated picture styles: surrealism, simulation of famous paintings, animation and comic styles, photography style simulation, black and white photography, old photo effects
# 

# ### Package import

# In[3]:


import os
import csv
import cv2
import numpy as np
import json
from sklearn import datasets
from pprint import pprint
from collections import namedtuple
from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


# ### Dataset Input 

# In[4]:


root='.'
label_path = os.path.join(root, 'images', 'annatation.jsonl')

#Create the Picture set class to store the information
class Pic_Inf:
    def __init__(self,idx=0,fname='',img=None,feat=None,label=None):
        self.idx = idx
        self.fname = fname
        self.img=img
        self.feat =feat
        self.label=label
        self.pred=None


flabels = []
ctr= 1

if os.path.exists(label_path):
    with open(label_path,encoding="utf8") as file:
        for line in file:
            try:
                json_object = json.loads(line)
                fname = f"{json_object['id']}.jpg"
                flabels.append((fname, json_object['annotation'], ctr))
                ctr +=1
            except json.JSONDecodeError:
                print(f"Error reading line {ctr}: not a valid JSON object")
                continue
else:
    raise ValueError('Invalid label file path [%s]'%label_path)

print(flabels)



# ### Pre-Process the Dataset

# In[20]:


images= []
labels= []

#fig, axs = plt.subplots(1, 4, figsize=(20, 5))  
for fname, label, idx in flabels:
    fpath = os.path.join(root, 'images', fname)

    if not os.path.isfile(fpath):
        print(f'{fpath} not found')
        continue
    
    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = cv2.resize(img, (128, 128))  


    images.append(img)
    labels.append(1 if label == 'realpic' else 0)
    
    
   
    '''
    H, W, C = img.shape
    ax = axs[(idx-1) % 4]
    ax.imshow(img)
    ax.set_title(f'{fname} - Label: {label}')
    ax.axis('off')  
    

    if (idx % 4 == 0) and (idx != len(flabels)):
        plt.show()
        fig, axs = plt.subplots(1, 4, figsize=(20, 5)) 
    '''


# ### Model Buiding

# In[6]:


from torch.functional import Tensor
from skimage import feature
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import time
import torch.nn as nn
import torch

import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset,Dataset
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

#images list store the image itself instead of the path

images = np.array(images)
labels = np.array(labels)

train_val_imgs, test_imgs, train_val_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=36)

# split to the train val and test
train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    train_val_imgs, train_val_labels, test_size=0.25, random_state=36)

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]



transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label


scaler = GradScaler()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_imgs = train_imgs.transpose((0, 3, 1, 2))
val_imgs = val_imgs.transpose((0, 3, 1, 2))
test_imgs = test_imgs.transpose((0, 3, 1, 2))

# TensorDataset
train_dataset = TensorDataset(torch.tensor(train_imgs, dtype=torch.float), torch.tensor(train_labels, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(val_imgs,dtype=torch.float), torch.tensor(val_labels, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(test_imgs,dtype=torch.float), torch.tensor(test_labels, dtype=torch.long))

# DataLoader
batch_size =  64  #ensure not over the size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size,num_workers=4)



# In[7]:


# define a classifier following the network
class classification_head(nn.Module):
    def __init__(self,in_ch,num_classes):    #need to change in the Bunus part
        super(classification_head,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_ch,num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

#define ResNet model
class Net(nn.Module):
    def __init__(self, input_ch, num_class,pretrained=True):
        super(Net,self).__init__()
        model = models.resnet50(pretrained=pretrained)
        self.backbone =  nn.Sequential(*list(model.children())[:-2])
        self.classification_head = classification_head(2048, num_class)

    def forward(self,x):
        x = self.backbone(x)
        output = self.classification_head(x)
        return output

# creat a model

model = Net(3, 2).to(device)        #change 2 -> 4 in Bunus Part

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

model.train()

# training

num_epochs = 300
train_losses = []  # average loss of each epoch

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, target in train_loader:
        inputs, target = inputs.to(device), target.to(device)
        
        optimizer.zero_grad()

         # Use automatic mixed precision
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, target)
        
           # Scale loss for backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)


    if (epoch+1) % 20 == 0:

        print('Epoch[{}/{}], loss: {:.6f}'
              .format(epoch+1, num_epochs, epoch_loss))
        


# ### Elavuate the Model

# In[16]:


model.eval()  
with torch.no_grad(): 
    correct = 0
    total = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on validation set: {100 * correct / total}%')


with torch.no_grad():  
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on Test set: {100 * correct / total}%')


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# #### comfusion Metrics

# In[9]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


cm = confusion_matrix(all_labels, all_preds)
print(cm)


plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# #### ROC and Auc change

# In[10]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


model.eval()  


all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)

       
        probs = torch.nn.functional.softmax(outputs, dim=1)
        all_preds.extend(probs[:, 1].cpu().numpy())  
        all_labels.extend(labels.cpu().numpy())



fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

'''
### ViT Part 

# In[61]:


! get_ipython().system(' pip install einops')
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)




# In[62]:


v = ViT(
    image_size=256,
    patch_size=32,
    num_classes=2,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

v = v.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(v.parameters(), lr=0.001)


num_epochs = 100
for epoch in range(num_epochs):
    for inputs, labels in train_loader:  
        inputs, labels = inputs.to(device), labels.to(device)
 
        optimizer.zero_grad()

        # forward
        outputs = v(inputs)
        loss = criterion(outputs, labels)

        # backward
        loss.backward()
        optimizer.step()

    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1, num_epochs, loss.item()))


# #### Evaluation of the Vit

# In[64]:


v.eval()  
with torch.no_grad():  
    correct = 0
    total = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = v(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on validation set: {100 * correct / total}%')


# ### Bonus:

# In[ ]:


flabels = []
ctr = 1

if os.path.exists(label_path):
    with open(label_path, encoding="utf8") as file:
        for line in file:
            try:
                json_object = json.loads(line)
                fname = f"{json_object['id']}.jpg"
                label = json_object['annotation']  
                flabels.append((fname, label, ctr))
                ctr += 1
            except json.JSONDecodeError:
                print(f"Error reading line {ctr}: not a valid JSON object")
                continue


# In[ ]:


class Net(nn.Module):
    def __init__(self, input_ch, num_class=5, pretrained=True): 
        super(Net, self).__init__()
        model = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.classification_head = classification_head(2048, num_class)

    def forward(self, x):
        x = self.backbone(x)
        output = self.classification_head(x)
        return output

'''