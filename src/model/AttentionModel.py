#imports for pytorch 
from model.layer.model import Attention
import torch
import torch.optim as optim
from torch.autograd import Variable


def trainModel(data, label, epochs, verbose = True, 
                                    lr=0.0005, 
                                    betas=(0.9, 0.999), 
                                    decay=0.0001, 
                                    fn = 256, 
                                    sn = 128, 
                                    tn = 64,
                                    dp = 0.2):
    # input size is taken as the first row of the first training data
    inputSize = len(data[0][0])
    model = Attention(inputSize, fn, sn, tn, dp)

    # create the optimizer 
    optimizer = optim.Adam(model.parameters(), 
                                    lr=lr, 
                                    betas=betas, 
                                    weight_decay=decay)

    for epoch in range(epochs):
        # set the model for train (turns on dropout)
        model.train()

        train_loss = 0. 
        train_error = 0. 
        for ibatch, batch in enumerate(data):
            # convert the data to pytorch readable data
            batch = Variable(torch.FloatTensor(batch))
            batchLabel = Variable(torch.LongTensor([label[ibatch]]))

            # set the gradiant to 0 for backward pass 
            optimizer.zero_grad()

            # forward pass
            loss, _ = model.calculate_objective(batch, batchLabel)
            train_loss += loss.data[0]
            error, _, _ = model.calculate_classification_error(batch, batchLabel)
            train_error += error

            #backward pass
            loss.backward()
            optimizer.step()

        #Calculate loss for the epoch
        train_loss /= len(data)
        train_error /= len(data)

        if verbose:
            print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, 
                                                                        train_loss.cpu().numpy()[0], 
                                                                        train_error))

    return model

def testModel(model, data, label):
    # stop the dropout 
    model.eval()

    # variables use to track error 
    test_loss = 0.
    test_error = 0.

    # create the array to store the instance weights and the 
    # bag probability and true label 
    attenArray = []
    probArray = []
    predictArray = []

    for ibatch, batch in enumerate(data):
        # create the variables
        batch = Variable(torch.FloatTensor(batch))
        batchLabel = Variable(torch.LongTensor([label[ibatch]]))

        loss, attention_weights = model.calculate_objective(batch, batchLabel)
        attenArray.append(attention_weights)
        test_loss += loss.data[0]

        error, predicted_label, prob = model.calculate_classification_error(batch, batchLabel)
        test_error += error

        probArray.append((prob.data.numpy()[0][0]).item())

        predictArray.append(int(predicted_label.cpu().data.numpy()[0][0]))

    return attenArray, probArray, predictArray

