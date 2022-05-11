from optim.LARS import LARS
import time
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

"""
this function is used for training models following SimCLR algorithm
"""

def train_simclr(train_loader, model, criterion, optimizer,epochs,batch_size,save,path=None):

    if optimizer == "Adam": #if optimizer type is ADAM, define ADAM, set lr 3e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    elif optimizer == "LARS": # if optimizer type is LARS, set LARS. those settings are from paper. 
         learning_rate = 0.3 * batch_size / 256
         optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-6,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )
        # "decay the learning rate with the cosine decay schedule without restarts"
         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 10, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    loss_list = [] #create empty loss list for training
    for epoch in range(epochs):
        stime = time.time() # start time
        loss_epoch = 0
        for step, ((x_i, x_j), _) in enumerate(train_loader):
            optimizer.zero_grad()
            x_i = x_i.cuda(non_blocking=True) #get first augmented imgs, load to GPU
            x_j = x_j.cuda(non_blocking=True) #get second augmented imgs, load to GPU

            h_i, h_j, z_i, z_j = model(x_i,x_j) #get embeddings and projections


            loss = criterion(z_i, z_j) #calculate NX-Tent loss
            loss.backward() #accumulate loss values

            optimizer.step()

            if optimizer == "LARS" and scheduler: # if LARS, update linear warm-up schduler
                scheduler.step()

            if step % 50 == 0 and step != 0: # print log in every 50 iterations
                print("Epoch: {}, step: {}/{}, loss: {}".format(epoch,step,len(train_loader),loss.item()))

            loss_epoch += loss.item() #accumulate loss values

        avg_loss_epoch = loss_epoch /len(train_loader) #calcualate average epoch loss 
        loss_list.append(avg_loss_epoch) # append average epoch loss to loss list 
        time_taken = (time.time()-stime)/60 # time end
        print("Epoch: {} completed, average loss: {}, time taken: {} mins".format(epoch,avg_loss_epoch,time_taken))

    if save:
        torch.save(model.state_dict(), path) # if save, save the model weights in input path
    return model,loss_list #returns model and loss list

"""
this function is used for downstream training for SimCLR models.
"""
def train_ds(train_loader, val_loader, model, criterion, optimizer,epochs,save,path=None):

    train_loss_list = [] #create empty train loss list for training
    train_acc_list = [] #create empty train acc for training
    val_loss_list = []  #create empty val loss list for training
    val_acc_list = [] #create empty val acc list for training
    for epoch in range(epochs):
        stime = time.time() # start time
        train_loss_epoch = 0 # init train_loss_epoch
        train_accuracy_epoch = 0 # init train_accuracy_epoch
        for step, (x, y) in enumerate(train_loader):
            model.train() # set model training mode
            optimizer.zero_grad()
            x = x.to(device = 'cuda:0', dtype = torch.float) #load to GPU
            y = y.to(device = 'cuda:0') #load to GPU

            preds = model(x) #make predictions for input images

            train_loss = criterion(preds, y) #calculate loss function

            predicted = preds.argmax(1) #get highest score class prediction
            train_acc = (predicted == y).sum().item() / y.size(0)
            train_accuracy_epoch += train_acc #update acc

            train_loss.backward() #accumulate loss values
            optimizer.step() #make update

            if step % 50 == 0 and step != 0:
                print("Epoch: {}, step: {}/{}, training_loss: {}, training_acc: {}".format(epoch,step,len(train_loader),train_loss.item(),train_acc))

            train_loss_epoch += train_loss.item()

        avg_trainloss_epoch = train_loss_epoch /len(train_loader)
        train_accuracy_epoch = train_accuracy_epoch/len(train_loader)
        train_loss_list.append(avg_trainloss_epoch)
        train_acc_list.append(train_accuracy_epoch)
        time_taken = (time.time()-stime)/60
        if val_loader == None:
            print("Epoch: {} completed, average train loss: {}, average training_acc: {} time taken: {} mins".format(epoch,avg_trainloss_epoch,train_accuracy_epoch,time_taken))

        if val_loader != None:

            model.eval() #if valloader is not none, calculate val loss and val acc. set model evaluation mode.
            with torch.no_grad():
                val_loss_epoch = 0
                val_accuracy_epoch = 0
                for step, (x, y) in enumerate(val_loader):
                    x = x.to(device = 'cuda:0', dtype = torch.float) # load to GPU
                    y = y.to(device = 'cuda:0') # load to GPU

                    preds = model(x)

                    val_loss = criterion(preds, y)

                    predicted = preds.argmax(1)
                    val_acc = (predicted == y).sum().item() / y.size(0)
                    val_accuracy_epoch += val_acc

                    #if step % 50 == 0 and step != 0:
                        #print("Epoch: {}, step: {}/{}, validation_loss: {}, validation_acc: {}".format(epoch,step,len(val_loader),val_loss.item(),val_acc))

                    val_loss_epoch += val_loss.item()

                avg_valloss_epoch = val_loss_epoch /len(val_loader)
                val_accuracy_epoch = val_accuracy_epoch/len(val_loader)
                val_loss_list.append(avg_valloss_epoch)
                val_acc_list.append(val_accuracy_epoch)
                time_taken = (time.time()-stime)/60
                print("Epoch: {} completed, average train loss: {}, average training_acc: {}, average validation loss: {}, average validation_acc: {} time taken: {} mins".format(epoch,avg_trainloss_epoch,train_accuracy_epoch,avg_valloss_epoch,val_accuracy_epoch,time_taken))

    if save:
        torch.save(model.state_dict(), path)

    if val_loader == None:
        return model,train_loss_list,train_acc_list,None,None

    return model,train_loss_list,train_acc_list,val_loss_list,val_acc_list

"""
this function is used for testing model performance of downstream simclr model.
"""
def test_ds(model, test_loader):
    model.eval() #set model in evaluation mode
    with torch.no_grad(): # no grad calculation
        test_acc = 0 # initialize test accuracy as 0
        stime = time.time() #start time
        for step, (x, y) in enumerate(test_loader):
            x = x.to(device = 'cuda:0', dtype = torch.float) #load to gpu
            y = y.to(device = 'cuda:0') #load to gpu

            output = model(x) #make prediction

            predicted = output.argmax(1)
            acc = (predicted == y).sum().item() / y.size(0) #calculate accc
            test_acc += acc #accumulate acc
 
        test_acc = test_acc/len(test_loader) #calculate average acc
        time_taken = (time.time()-stime)/60 #finish time
        print("Test accuracy: {}, time taken: {}".format(test_acc,time_taken)) # print test acc
