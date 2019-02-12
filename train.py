'''
File: train.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Sunday, 2018-11-18 14:42
Last Modified: Sunday, 2018-11-18 14:43
--------------------------------------------
Desscription:
'''


import torch
import torchvision.transforms as transforms
import os 
import sys
import pickle
import numpy as np
import nltk
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from generate_vocab_dict import Vocabulary
from data_loader import CocoDataset, coco_batch
from pycocotools.coco import COCO
import argparse
# from model_V2 import Encoder, Decoder
from model_V2_dropout0 import Encoder, Decoder
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn



# Device configuration


def main():
    # Create model directory
    ##### arguments #####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    PATH = os.getcwd()
    image_dir = '/u/training/tra379/final_project/resized_2014/'
    caption_path = '/u/training/tra379/final_project/captions_train2014.json'
    vocab_path = '/u/training/tra379/final_project/data/vocab_self_new.pkl'
    model_path = '/u/training/tra379/scratch/model_layer_10'
    crop_size = 224
    batch_size = 128
    num_workers = 4
    learning_rate = 0.001

    # Decoder
    embed_size = 512
    hidden_size = 512
    num_epochs = 5
    log_step = 100
    save_step = 1000
    ######################

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    coco = CocoDataset(image_dir, caption_path, vocab, transform)
    dataLoader = torch.utils.data.DataLoader(coco, batch_size, shuffle=True, num_workers=4, collate_fn=coco_batch)
    num_layers=10
    print("Number of Layer: ", num_layers)
    # Declare the encoder decoder
    encoder = Encoder(embed_size=embed_size).to(device)
    decoder = Decoder(embed_size=embed_size, hidden_size=hidden_size, vocab_size=len(vocab), num_layers=num_layers, stateful=False).to(device)
    #encoder.load_state_dict(torch.load('/u/training/tra379/final_project/models_self/encoder-2-2000.ckpt'))
    #decoder.load_state_dict(torch.load('/u/training/tra379/final_project/models_self/decoder-2-2000.ckpt'))
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.resnet.fc.parameters())
    # For encoder only train the last fc layer
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Train the models
    total_step = len(dataLoader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(dataLoader):

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if('step' in state and state['step']>=1024):
                        state['step'] = 1000
            optimizer.step()

            # Print log info
            if i % log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

            # Save the model checkpoints
            if (i+1) % save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    model_path, 'decoder-{}-{}-{}.ckpt'.format(num_layers, epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    model_path, 'encoder-{}-{}-{}.ckpt'.format(num_layers, epoch+1, i+1)))
    torch.save(decoder, os.path.join(model_path, 'decoder_final-{}.model'.format(num_layers)))
    torch.save(encoder, os.path.join(model_path, 'encoder_final-{}.model'.format(num_layers)))

if __name__ == "__main__":
    print('Start Training')
    main()
