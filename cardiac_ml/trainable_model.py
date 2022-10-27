"""Trainable model defines the interface to train a model"""

import torch
import time
import copy
import tqdm

class TrainableModel:
    """TrainableModel class.

    Define the evalaute and learn methods to train a Neural Network model.
    """

    def __init__(self, criterion, optimizer, scheduler, outputHandler, device="cpu", progressbar = True):
        """Constructor"""
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.outputHandler = outputHandler
        self.progressbar = progressbar

    def learn(self, model, trainingData, valData, num_epochs, grad_clippling=False, checkpointRate = None, name = "mymodel"):
        """Training loop method"""
        if checkpointRate is None:
            checkpointRate = int(num_epochs/10)

        # Intial time
        since = time.time()

        bestLoss = 1e30
        bestEpoch = -1
        bestWeights = copy.deepcopy(model.state_dict())

        with tqdm.tqdm(range(num_epochs), unit="epoch", disable = not self.progressbar) as progEpoch:
        
            # Name of status bar in terminal
            progEpoch.set_description("Learning")

            # Mini-batch gradient descent: num_epochs loop over the whole training set
            for epoch in progEpoch:
                # Adjust the learning rate based on the number of epochs through the provided scheduler
                self.scheduler.step()

                # Mini-batch gradient descent: num_batches loop over batches of training examples
                trainingLoss = self.evaluate(model, trainingData, training=True, grad_clippling=grad_clippling, description="Training")
                validationLoss = self.evaluate(model, valData, training=False, grad_clippling=False, description="Validate")

                if validationLoss < bestLoss:
                    bestLoss = validationLoss
                    bestEpoch = epoch
                    bestWeights = copy.deepcopy(model.state_dict())
                    # Overwrite best model
                    self.checkpoint(model,'{}.best.pth'.format(name))

                if (epoch % checkpointRate) == 0 :
                    self.checkpoint(model,"{}.training.epoch.{}.pth".format(name,epoch))

                # Online stats in terminal 
                tqdm.tqdm.write("Epoch {:3d}/{},  training : {:11.4f}   validation : {:11.4f}   best validation loss : {:11.4f} on epoch {}"
                                .format(epoch + 1 , num_epochs, trainingLoss, validationLoss, bestLoss, bestEpoch + 1))
                
                # Print stats into file                                          
                self.outputHandler.write_errors(epoch + 1, num_epochs, trainingLoss, validationLoss, bestLoss, bestEpoch + 1 )
                self.outputHandler.write_grads(model.parameters())    

        timeElapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(timeElapsed // 60, timeElapsed % 60))
        print('Best val Loss: {:.4f}'.format(bestLoss))

        model.load_state_dict(bestWeights)
        self.checkpoint(model,'{}.final.best.bestEpoch.{}.pth'.format(name, bestEpoch))

    # Evaluate method: performs either training or simple evaluation depending on wether it is training set or the validation set
    def evaluate(self, model, dataloader, training=False, grad_clippling=False, description=""):

        # Tell the model that wether you are training the model or simply testing it.
        # Effectively layers like dropout, batchnorm etc. which behave different on the train and test
        # procedures know what is going on and hence can behave accordingly.
        if training:
            model.train()
        else:
            model.eval()

        running_loss = 0
        # Lenght of dataloader == Numbers of batches in which the data set is splitted
        num_batches = len(dataloader)

        with tqdm.tqdm(dataloader, unit=" batches", disable = not self.progressbar) as progData:  

            # Name of status bar in terminal
            progData.set_description(description)

            # Mini-batch gradient descent: loop over batches of training examples
            # inputs.size() = (batch_size, D_in)
            # labels.size() = (batch_size, D_out)
            for inputs, labels in progData:
                
                progData.set_postfix(loss="{:4f}".format(running_loss/num_batches))
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                with torch.set_grad_enabled(training):
                    # Forward pass: compute predicted outputs by passing inputs to the model.
                    outputs = model(inputs)
                    # Compute loss.
                    loss = self.criterion(outputs, labels)

                    if training:
                        # Backward pass: compute gradient of the loss with respect to model parameters
                        loss.backward()
                        # Gradient clipplig to prevent exploding gradients
                        if grad_clippling:
                            torch.nn.utils.clip_grad_norm_((rnn_param for rnn_param in list(model.parameters())[:-2]), 10, norm_type = 2)
                        # Calling the step function on an Optimizer makes an update to its parameters
                        self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / num_batches
        return epoch_loss

    # Save model with name
    def checkpoint(self,model,name):
        torch.save(model, name)
