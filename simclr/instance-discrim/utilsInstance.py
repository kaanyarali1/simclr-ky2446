import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


"""
this function is used for training instance-level classification. It returns the model, losslist and memorybank after
completing all iterations. Also, you can save the model in input path.
"""
def trainInstance(model,epochs,train_loader,criterion,optimizer,save,path_model=None,path_mem=None):
  loss_list = []
  for epoch in range(epochs):
    stime = time.time() #start time
    epoch_loss = 0 #initialize epoch loss 
    for i, (input, _, index) in enumerate(train_loader):
      optimizer.zero_grad() 
      imgs = torch.FloatTensor(input).cuda() #load image to GPU
      labels = index.cuda() # you do not need label but you need INDEX of images
      output, memory = model(imgs,index) # pass the input image through model and get output and memory bank
      #feature = feature.cuda()
      #output = lemniscate(feature, index)
      output = output.cuda() # load output to GPU
      #return output
      loss = criterion(output, index) #calculate loss
      epoch_loss  += loss.item() #update epoch loss
      loss.backward() #accumulate loss values
      optimizer.step() #make an update
      #return output
      model.updateMemory(memory) #update the memory bank at the end of each iteration
      if i % 50 == 0 and i != 0:
        print("Epoch: {}, step: {}/{}, loss: {}".format(epoch,i,len(train_loader),loss.item())) # print some log at every 50 iterations
    avg_loss_epoch = epoch_loss / len(train_loader) #calculate average epoch loss
    loss_list.append(avg_loss_epoch) # append average epoch loss to list
    time_taken = (time.time()-stime)/60 # calculate time spent on iteration
    print("Epoch: {} completed, average loss: {}, time taken: {} mins".format(epoch,avg_loss_epoch,time_taken)) 
  if save: # if save is True, save model weights and memory bank in input path
    torch.save(memory,path_mem)
    torch.save(model.state_dict(),path_model)
  return model,loss_list,model.memory

"""
this function builds a knn classifier based on the memory bank values and its corresponding labels. set hyperparameter K
"""
def build_knn(memomory,training_labels,K):
  memomory = memomory.cpu().detach().numpy() # load to cpu and cast as np array
  training_labels = training_labels.cpu().detach().numpy()  # load to cpu and cast as np array
  neigh = KNeighborsClassifier(n_neighbors=K) #build knn classifier with input K value
  neigh.fit(memomory, training_labels) 
  return neigh

"""
this function is used for prediction of test images using knn classifier. input: model, test_loaderi and knn classifier
returns predicted testlabels from knn classifier and ground truth labels
"""
def knn_predict(test_loader,knn_classifier,model):
  predicted_test_labels = [] #create empty list for predictions
  labels = [] 
  model.eval() # set the model evaluation mode
  for (x1 , y) in (test_loader): # for all mini-batches in the test set
    x1 = x1.to(device = 'cuda:0', dtype = torch.float) #load images to GPU
    features = model.getEmbedding(x1) # pass input through model and get its feature embedding
    #print(features.shape)
    features = features.cpu().detach().numpy() #load to cpu
    predicted_classes = knn_classifier.predict(features) # make prediction for this embedding
    predicted_test_labels.append(predicted_classes) # append the predictions to the list
    labels.append(y) # append the ground truth labels to this list. this will be used for final accuracy calculation
  predicted_test_labels = np.concatenate(predicted_test_labels, axis=0) # reshape size
  labels = np.concatenate(labels, axis=0) # reshape size
  return predicted_test_labels, labels

"""
it returns the accuracy for given predicted labels
"""
def get_accuracy(labels,predicted_test_labels):
  test_accuracy = accuracy_score(labels,predicted_test_labels)
  return test_accuracy

"""
this function is used for plotting k nearest neighbours of the given input image in the memory bank
"""
def plotNeighbour(resnet,knn_classifier,test_img,training_images,row=1, cols=6):
  test_img = test_img.reshape(1,3,32,32) # reshape test image to (3,32,32) -> (1,3,32,32)
  test_img = test_img.to(device = 'cuda:0', dtype = torch.float) #load to GPU
  input_embedding = resnet.getEmbedding(test_img).cpu().detach() #get feature embedding from the model 
  test_img = test_img.cpu().detach() #load img to cpu back
  dist, ind = knn_classifier.kneighbors(input_embedding.reshape(1,2048)) # calculate the nearest neighbours of the embedding in the memory bank
  close = torch.cat((test_img,training_images[ind])) # select closest images in the trainin set using indexing
  fig = plt.figure(figsize=(40,10))
  for i in range(close.shape[0]):
    ax = fig.add_subplot(row, cols, i+1, xticks=[], yticks=[]) #plot images
    ax.imshow(close[i].permute(1, 2, 0))
  plt.show()