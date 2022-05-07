from optim.LARS import LARS
import time
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def train_simclr(train_loader, model, criterion, optimizer,epochs,batch_size,save,path=None):

    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    elif optimizer == "LARS":
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

    loss_list = [] 
    for epoch in range(epochs):
        stime = time.time()
        loss_epoch = 0
        for step, ((x_i, x_j), _) in enumerate(train_loader):
            optimizer.zero_grad()
            x_i = x_i.cuda(non_blocking=True)
            x_j = x_j.cuda(non_blocking=True)

            # positive pair, with encoding
            h_i, h_j, z_i, z_j = model(x_i,x_j)


            loss = criterion(z_i, z_j)
            loss.backward()

            optimizer.step()

            if optimizer == "LARS" and scheduler:
                scheduler.step()

            if step % 50 == 0 and step != 0:
                print("Epoch: {}, step: {}/{}, loss: {}".format(epoch,step,len(train_loader),loss.item()))

            loss_epoch += loss.item()

        avg_loss_epoch = loss_epoch /len(train_loader)
        loss_list.append(avg_loss_epoch)
        time_taken = (time.time()-stime)/60
        print("Epoch: {} completed, average loss: {}, time taken: {} mins".format(epoch,avg_loss_epoch,time_taken))

    if save:
        torch.save(model.state_dict(), path)
    return model,loss_list

def train_ds(train_loader, val_loader, model, criterion, optimizer,epochs,save,path=None):

    train_loss_list = [] 
    train_acc_list = []
    val_loss_list = [] 
    val_acc_list = []
    for epoch in range(epochs):
        stime = time.time()
        train_loss_epoch = 0
        train_accuracy_epoch = 0
        for step, (x, y) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            x = x.to(device = 'cuda:0', dtype = torch.float)
            y = y.to(device = 'cuda:0')

            preds = model(x)

            train_loss = criterion(preds, y)

            predicted = preds.argmax(1)
            train_acc = (predicted == y).sum().item() / y.size(0)
            train_accuracy_epoch += train_acc

            train_loss.backward()
            optimizer.step()

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

            model.eval()
            with torch.no_grad():
                val_loss_epoch = 0
                val_accuracy_epoch = 0
                for step, (x, y) in enumerate(val_loader):
                    x = x.to(device = 'cuda:0', dtype = torch.float)
                    y = y.to(device = 'cuda:0')

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



def test_ds(model, test_loader):
    model.eval()
    with torch.no_grad():
        test_acc = 0
        stime = time.time()
        for step, (x, y) in enumerate(test_loader):
            x = x.to(device = 'cuda:0', dtype = torch.float)
            y = y.to(device = 'cuda:0')

            output = model(x)

            predicted = output.argmax(1)
            acc = (predicted == y).sum().item() / y.size(0)
            test_acc += acc

        test_acc = test_acc/len(test_loader)
        time_taken = (time.time()-stime)/60
        print("Test accuracy: {}, time taken: {}".format(test_acc,time_taken))


def plot_features(model, data_loader, images, labels, num_classes, num_feats, batch_size):
    preds = np.array([]).reshape((0,1))
    gt = np.array([]).reshape((0,1))
    feats = np.array([]).reshape((0,num_feats))
    model.eval()
    with torch.no_grad():
        for (x1 ,_) in (data_loader):
            x1 = x1.squeeze().to(device = 'cuda:0', dtype = torch.float)
            out = model(x1)
            out = out.cpu().data.numpy()#.reshape((1,-1))
            feats = np.append(feats,out,axis = 0)
    
    tsne = TSNE(n_components = 2, perplexity = 50)
    x_feats = tsne.fit_transform(feats)
    num_samples = int(batch_size*(images.shape[0]//batch_size))#(len(val_df)
    
    for i in range(num_classes):
        plt.scatter(x_feats[labels[:num_samples]==i,1],x_feats[labels[:num_samples]==i,0])
    
    plt.legend([str(i) for i in range(num_classes)])
    plt.title("TSNE - Test Set")
    plt.show()