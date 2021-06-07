import torch
from PIL import Image
import time
import torch.nn as nn
from torchvision import transforms, datasets, models
from tqdm import tqdm
import numpy as np


class myresize:
    def __init__(self, size) -> None:
        self.size = size
    def __call__(self, pic):
        w, h = pic.size
        return pic.resize((self.size, int(self.size*h/w)))
    def __repr__(self):
        return self.__class__.__name__ + '()'

preprocess = transforms.Compose([
    myresize(224),
    transforms.Pad(padding=224, fill=255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Grayscale(3),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class eyeNet(nn.Module):
    def __init__(self, ckpt=None):
        super().__init__()
        self.base_encoder = models.resnet18(pretrained=False, progress=True, num_classes=2)
        #self.base_encoder = torch.hub.load('/Users/sunxinyu/.cache/torch/hub/vision-0.9.0', 
        #                                   'squeezenet1_1', pretrained=True, source='local')
        #self.pool = nn.AdaptiveMaxPool2d((1, 1))
        #self.fc = nn.Linear(512, 2)
        if ckpt:
            self.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))
        
    def forward(self, input):
        feat = self.base_encoder(input)
        prob = nn.functional.softmax(feat, dim=1)
        return prob

    @torch.no_grad()
    def inference(self, input):
        self.eval()
        res = self.forward(input)
        return torch.argmax(res, dim=1)
    

class context():
    def __init__(self, model, num_epoch, lr):
        self.num_epochs = num_epoch
        self.learning_rate = lr
        self.model = model
        ds = datasets.ImageFolder('dataset', preprocess)
        print(ds.class_to_idx)
        self.trainloader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)
        self.critirion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.num_epochs,
            eta_min=self.learning_rate / 1000
        )


    def train(self):
        for epoch in range(self.num_epochs):
            loss, acc = self.train_epoch(epoch)
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            print(f'epoch:{epoch} loss:{loss} acc:{acc} lr:{lr}')

    def train_epoch(self, epoch):
        self.model.train()

        loss_meter = []
        acc_meter = []
        
        for index, (images, labels) in enumerate(self.trainloader):
            outputs = self.model(images)
            loss = self.critirion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.append(loss.item())
            acc_meter.append(binary_accuracy(outputs, labels))
            print(f"epoch:{epoch}\tstep:{index}\tloss:{loss.item()}\tacc:{acc_meter[-1]}")
            
        loss_meter = np.asarray(loss_meter)
        loss_epoch = loss_meter.sum() / len(loss_meter)
        acc_meter = np.asarray(acc_meter)
        acc_epoch = acc_meter.sum() / len(acc_meter)

        return loss_epoch, acc_epoch


def inference(model):
    model = model.eval()

    input = Image.open('dataset/open/eye_162295761332.png').convert('RGB')
    input = preprocess(input)
    input = input.unsqueeze(0)

    tic = time.time()
    
    res = model.inference(input)

    print(res)

    toc = time.time() - tic
    print(f'use time: {toc * 1000}ms')


def binary_accuracy(output, target, threshold=0.5):
    batch_size = target.shape[0]
    pred = torch.argmax(output, dim=1)
    correct = pred.eq(target).sum()
    return correct * (100.0 / batch_size)


if __name__ == '__main__':
    model = eyeNet()
    
    inference(model)


    