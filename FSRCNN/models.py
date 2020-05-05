import math
import torch
from torch import nn
import cv2
from net_canny import Net
from torch.autograd import Variable
from utils import convert_ycbcr_to_rgb, preprocess
import numpy as np


def WeightedLoss(preds, labels, label_edges):
    # preds [batch_size, 1, 24, 24]
    # labels [batch_size, 1, 24, 24]
    
    batch_size = labels.shape[0]
    loss = 0
    # get edge from labels
    for i in range(batch_size):
        label = labels[i,:,:,:]
        pred = preds[i,:,:,:]
        label_edge = label_edges[i,:,:,:] 
        
        # convert to array
        #pred = pred.detach().cpu().numpy().squeeze(0)#.squeeze(0)
        #label = label.detach().cpu().numpy().squeeze(0)#.squeeze(0)
        #label_edge = label_edge.detach().cpu().numpy().squeeze(0)
        
        # use edge as weight of loss
        #pred_cal = pred*label_edge
        loss += torch.norm(label_edge*pow(pred - label, 2))
        #loss += torch.mean(label_edge*abs(pred - label))
        
    loss /= batch_size
    
    #return torch.tensor(loss, requires_grad=True)
    return loss

def MSCE(preds, labels, label_edges):
    lamda = 0.85
    batch_size = labels.shape[0]
    l_mse = 0
    l_edge = 0
    
    #net = Net(threshold=1.0, use_cuda=True)
    #net.cuda()
    #net.eval()
    
    for i in range(batch_size):
        label_edge = label_edges[i,:,:,:]
        label = labels[i,:,:,:]
        pred = preds[i,:,:,:]
        
        # reshape to [1,1,160,160]
        # label = label.view(1,1,160,160)
        # pred = pred.view(1,1,160,160)
        # label = Variable(label)
        # pred = Variable(pred)
        
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # get cbcr from lr
        #_, ycbcr = preprocess(lr, device)
        #ycbcr = convert_ycbcr_to_rgb(lr)
        
        # convert ycbcr to rgb
        #red = pred.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        #pred = np.array([pred, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        #pred = np.clip(convert_ycbcr_to_rgb(pred), 0.0, 255.0).astype(np.uint8)
        #output = pil_image.fromarray(output)
        
        # extract edge
        #pred_edge = cv2.Canny(pred, 50,200)
        #pred_edge = canny(pred)
        
        # calculate loss
        l_mse += torch.norm(pred - label) 
        l_edge += torch.norm(label_edge*pred - label_edge*label)
    
    loss = lamda*l_mse + (1-lamda)*l_edge
    loss /= batch_size
    
    return loss


class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
