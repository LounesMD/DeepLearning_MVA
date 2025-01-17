import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
from torchvision import transforms
import time
import os 
from tqdm import tqdm
from models.models import ConvLSTM,PhyCell, EncoderRNN
from data.moving_mnist import MovingMNIST
from constrain_moments import K2M
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
import argparse

from utils import ImageDataset, get_images_from_path

device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='data/')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--nepochs', type=int, default=2001, help='nb of epochs')
parser.add_argument('--print_every', type=int, default=1, help='')
parser.add_argument('--eval_every', type=int, default=1, help='')
parser.add_argument('--save_name', type=str, default='phydnet', help='')
args = parser.parse_args()

# Path to the folder containing images
train_dataset = get_images_from_path('./weather_dataset0/train')
train_dataset = ImageDataset(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = get_images_from_path('./weather_dataset0/test')
test_dataset = ImageDataset(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

val_dataset = get_images_from_path('./weather_dataset0/validation')
val_dataset = ImageDataset(val_dataset)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

constraints = torch.zeros((49,7,7)).to(device)
ind = 0
for i in range(0,7):
    for j in range(0,7):
        constraints[ind,i,j] = 1
        ind +=1    

def train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion,teacher_forcing_ratio, double_lstm = False):                
    encoder_optimizer.zero_grad()
    # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
    input_length  = input_tensor.size(1)
    target_length = target_tensor.size(1)
    loss = 0
    for ei in range(input_length-1): 
        encoder_output, encoder_hidden, output_image,_,_ = encoder(input_tensor[:,ei,:,:,:].float(), (ei==0) )
        loss += criterion(output_image,input_tensor[:,ei+1,:,:,:])

    decoder_input = input_tensor[:,-1,:,:,:] # first decoder input = last image of input sequence
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 
    for di in range(target_length):
        decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input.float())
        target = target_tensor[:,di,:,:,:]
        loss += criterion(output_image,target)
        if use_teacher_forcing:
            decoder_input = target # Teacher forcing    
        else:
            decoder_input = output_image

    if not double_lstm:
        # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
        k2m = K2M([7,7]).to(device)
        for b in range(0,encoder.phycell.cell_list[0].input_dim):
            filters = encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)     
            m = k2m(filters.double()) 
            m  = m.float()   
            loss += criterion(m, constraints) # constrains is a precomputed matrix   
        
    loss.backward()
    encoder_optimizer.step()
    return loss.item() / target_length


def trainIters(encoder, nepochs, print_every=10,eval_every=10,name='', double_lstm=False):
    train_losses = []
    best_mse = float('inf')

    encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=0.001)
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=2,factor=0.1,verbose=True)
    criterion = nn.MSELoss()
    
    for epoch in tqdm(range(0, nepochs)):
        t0 = time.time()
        loss_epoch = 0
        teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003) 
        
        for i, out in tqdm(enumerate(train_loader, 0)):
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)
            loss = train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion, teacher_forcing_ratio, double_lstm=double_lstm)                                   
            loss_epoch += loss
                      
        train_losses.append(loss_epoch)        
        if (epoch+1) % print_every == 0:
            print('epoch ',epoch,  ' loss ',loss_epoch, ' time epoch ',time.time()-t0)
            
        if (epoch+1) % eval_every == 0:
            mse, mae,ssim = evaluate(encoder,test_loader) 
            scheduler_enc.step(mse)                   
            torch.save(encoder.state_dict(),'save/encoder_{}.pth'.format(name))                           
    return train_losses

    
def evaluate(encoder,loader):
    total_mse, total_mae,total_ssim,total_bce = 0,0,0,0
    t0 = time.time()
    with torch.no_grad():
        for i, out in enumerate(loader, 0):
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)
            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]

            for ei in range(input_length-1):
                encoder_output, encoder_hidden, _,_,_  = encoder(input_tensor[:,ei,:,:,:].float(), (ei==0))

            decoder_input = input_tensor[:,-1,:,:,:] # first decoder input= last image of input sequence
            predictions = []

            for di in range(target_length):
                decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input.float(), False, False)
                decoder_input = output_image
                predictions.append(output_image.cpu())

            input = input_tensor.cpu().numpy()
            target = target_tensor.cpu().numpy()
            predictions =  np.stack(predictions) # (10, batch_size, 1, 64, 64)
            predictions = predictions.swapaxes(0,1)  # (batch_size,10, 1, 64, 64)
            # Assume predictions and target have shapes as described: (16, 10, 1, 64, 64)
            # Select the first batch
            predictions_batch = predictions[0]  # Shape: (10, 1, 64, 64)
            target_batch = target[0]  # Shape: (10, 1, 64, 64)

            # Set up the figure
            fig, axes = plt.subplots(2, 5, figsize=(20, 5))

            # Plot target images in the first row
            for i in range(5):
                axes[0, i].imshow(target_batch[i, 0], cmap='gray')  # Take the first channel
                axes[0, i].axis('off')  # Hide axes
                axes[0, i].set_title(f'Target {i+1}')  # Title for each column

            # Plot prediction images in the second row
            for i in range(5):
                axes[1, i].imshow(predictions_batch[i, 0], cmap='gray')  # Take the first channel
                axes[1, i].axis('off')  # Hide axes
                axes[1, i].set_title(f'Pred {i+1}')  # Title for each column

            plt.tight_layout()
            plt.show()

            # breakpoint()

            mse_batch = np.mean((predictions-target)**2 , axis=(0,1,2)).sum()
            mae_batch = np.mean(np.abs(predictions-target) ,  axis=(0,1,2)).sum() 
            total_mse += mse_batch
            total_mae += mae_batch
            
            for a in range(0,target.shape[0]):
                for b in range(0,target.shape[1]):
                    total_ssim += ssim(target[a,b,0,], predictions[a,b,0,], data_range=1.0) / (target.shape[0]*target.shape[1]) 

            
            cross_entropy = -target*np.log(predictions) - (1-target) * np.log(1-predictions)
            cross_entropy = cross_entropy.sum()
            cross_entropy = cross_entropy / (args.batch_size*target_length)
            total_bce +=  cross_entropy
     
    print('eval mse ', total_mse/len(loader),  ' eval mae ', total_mae/len(loader),' eval ssim ',total_ssim/len(loader), ' time= ', time.time()-t0)        
    return total_mse/len(loader),  total_mae/len(loader), total_ssim/len(loader)


#  phycell  =  PhyCell(input_shape=(16,16), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device)
phycell  =  PhyCell(input_shape=(16,16), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device)
convcell =  ConvLSTM(input_shape=(16,16), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device=device)   
encoder  = EncoderRNN(phycell, convcell, device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
   
print('phycell ' , count_parameters(phycell) )    
print('convcell ' , count_parameters(convcell) ) 
print('encoder ' , count_parameters(encoder) ) 
print(encoder)

encoder.load_state_dict(torch.load('save/encoder_phydnet_weather_200.pth', map_location=torch.device('cpu')))
encoder.eval()
mse, mae,ssim = evaluate(encoder,test_loader) 