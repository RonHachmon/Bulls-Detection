
import os
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import matplotlib.pyplot as plt


from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

# setting device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#create network
net = create_mobilenetv1_ssd(2)
config = mobilenetv1_ssd_config


# setting images with  pre processing inputs
train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
target_transform = MatchPrior(config.priors, config.center_variance,
                                config.size_variance, 0.5)
test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)


#load train dataset and save labels

#require data set in format of open images in data folder to run
dataset_path='data/open_images'
label_path = 'data'
datasets = []
dataset = OpenImagesDataset(dataset_path,
                 transform=train_transform, target_transform=target_transform,
                 dataset_type="train", balance_data=True)

# store labels
label_file = os.path.join(label_path, "open-images-model-labels.txt")
with open(label_file, "w") as f:
    f.write("\n".join(dataset.class_names))
    
datasets.append(dataset)


train_dataset = ConcatDataset(datasets)
train_loader = DataLoader(train_dataset,batch_size=24,
                          num_workers=2,
                          shuffle=True)

#load validation dataset 
val_dataset = OpenImagesDataset(dataset_path,
                  transform=test_transform, target_transform=target_transform,
                  dataset_type="test")

val_loader  = DataLoader(val_dataset,batch_size=24,
                          num_workers=2,
                          shuffle=False)


def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False

# freeze all layers except the last one
freeze_net_layers(net.base_net)
freeze_net_layers(net.source_layer_add_ons)
freeze_net_layers(net.extras)
params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())



# load pre trained model
net.init_from_pretrained_ssd('Transfer-Learing-SSD-model/models/mobilenet-v1-ssd-mp-0_675.pth')

# loading model
net.to(DEVICE)


#creating new loss function and optimzer
criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                          center_variance=0.1, size_variance=0.2, device=DEVICE)

optimizer = optim.Adam(params)

scheduler = CosineAnnealingLR(optimizer, 120, last_epoch=-1)

def train(loader, net, criterion, optimizer, device):
    net.train(True)
    for  data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes) 
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()



def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

val_loss_arr = []
val_regression_loss_arr = []
val_classification_loss_arr = []
val_loss=0
val_regression_loss=0
val_classification_loss=0
my_range=100
for epoch in range(my_range):
    scheduler.step()
    train(train_loader, net, criterion, optimizer,
          device=DEVICE)
    
    
    #measure learning curve:
    val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
    val_loss_arr.append(val_loss)
    val_regression_loss_arr.append(val_regression_loss)
    val_classification_loss_arr.append(val_classification_loss)



# save model
model_path = os.path.join('models', f"ssd-mb1-{my_range}-Loss-{val_loss}.pth")
net.save(model_path)



# displaying the model learning rate 
plt.plot(val_classification_loss_arr)
plt.xlabel('epoch')
plt.ylabel('val classification loss')
plt.title("val classification loss")
plt.show()



