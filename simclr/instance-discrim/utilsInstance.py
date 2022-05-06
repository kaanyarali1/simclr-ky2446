import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



def trainInstance(model,epochs,train_loader,lemniscate,criterion,optimizer,save,path_model=None,path_mem=None):
  loss_list = []
  for epoch in range(epochs):
    stime = time.time()
    epoch_loss = 0
    for i, (input, _, index) in enumerate(train_loader):
      optimizer.zero_grad()
      imgs = torch.FloatTensor(input).cuda()
      labels = index.cuda()
      feature = model(imgs)
      feature = feature.cuda()
      output = lemniscate(feature, index)
      loss = criterion(output, index)
      epoch_loss  += loss.item()
      loss.backward()
      optimizer.step()
      if i % 50 == 0 and i != 0:
        print("Epoch: {}, step: {}/{}, loss: {}".format(epoch,i,len(train_loader),loss.item()))
    avg_loss_epoch = epoch_loss / len(train_loader)
    loss_list.append(avg_loss_epoch)
    time_taken = (time.time()-stime)/60
    print("Epoch: {} completed, average loss: {}, time taken: {} mins".format(epoch,avg_loss_epoch,time_taken))
  if save:
    torch.save(lemniscate.memory,path_mem)
    torch.save(model.state_dict(),path_model)
  return model,avg_loss_epoch,lemniscate.memory

def build_knn(memomory,training_labels,K):
  neigh = KNeighborsClassifier(n_neighbors=K)
  neigh.fit(memomory, training_labels)
  return neigh


def knn_predict(test_loader,knn_classifier,model):
  predicted_test_labels = []
  labels = []
  model.eval()
  for (x1 , y) in (test_loader):
    x1 = x1.to(device = 'cuda:0', dtype = torch.float)
    features = model(x1)
    features = features.cpu().detach().numpy() 
    predicted_classes = knn_classifier.predict(features)
    predicted_test_labels.append(predicted_classes)
    labels.append(y)
  predicted_test_labels = np.concatenate(predicted_test_labels, axis=0)
  labels = np.concatenate(labels, axis=0)
  return predicted_test_labels, labels

def get_accuracy(labels,predicted_test_labels):
  test_accuracy = accuracy_score(labels,predicted_test_labels)
  return test_accuracy