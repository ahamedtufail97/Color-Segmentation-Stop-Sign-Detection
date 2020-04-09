#author @Siddarth
from typing import Any
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import expit #Importing logistic sigmoid function
import sklearn.metrics as sklm
import os #Imported necesary modules

def init(dimension):
    w_init = np.random.randn(dimension, 1)
    lr = 0.01 #learning rate
    eps = 1e-4
    epochcount = 50 #Total number of epochs
    return w_init, lr, eps, epochcount


def safe_log(z, minimumvalue=0.0000000001): #referred from internet
    return np.log(z.clip(min=minimumvalue))


def lossfn(X, y, w):
    z = np.matmul(X, w) #matrix multiplication
    sigmaofz = expit(z)
    v1 = y * safe_log(sigmaofz)
    v2 = (1 - y) * safe_log(1 - sigmaofz)
    v = v1 + v2
    ret = v.sum()
    return -1 * ret


def gradient_derivative(X, y, w):
    in_sigma = np.matmul(X, w)
    sig = expit(in_sigma)  #Logistic sigmoid function
    co_efficient = y - sig
    grady = co_efficient * X #Gradient = grady :-)
    summer = np.sum(grady, axis=0)
    ret = np.reshape(summer, w.shape)
    return ret


def state_saving(w, l, e):
    save_path = "./output/logistic_basic/"
    wei_name = "weights_" + str(e)
    l_name = "loss_" + str(e)
    np.save(save_path + wei_name + ".npy", w)
    np.save(save_path + l_name + ".npy", l)
    np.savetxt(save_path + wei_name + ".txt", w)
    with open(save_path + l_name + ".txt", "w") as f:
        f.write(str(l))

def main():
    # the path for training set
    train_img_path = "./data/train/images/"
    train_mask_path = "./data/train/masks/"

    # the path for validation set
    val_img_path = "./data/val/images/"
    val_mask_path = "./data/val/masks/"

    # read one image to initialize values
    samp = np.load(train_img_path + "2.npy")
    num_channel = samp.shape[2]

    # initial weight(w), learning rate(lr) and threshold(eps)
    w, lr, eps, epochcount = init(num_channel + 1)
    print(w.shape, w.dtype, np.min(w), np.max(w))
    print(lr, eps, epochcount)

    sum_loss = 0
    avg_loss = 0
    step = 0

    # training code below:
    for e in range(epochcount):
        print("In epoch number:", e + 1)
        for file_name in os.listdir(train_img_path):
            X = np.load(os.path.join(train_img_path, file_name))
            X = np.reshape(X, (X.shape[0] * X.shape[1], 3))
            o = np.ones((X.shape[0], 1))
            X = np.concatenate((o, X), 1)
            y = np.load(os.path.join(train_mask_path, file_name))
            y = np.reshape(y, (y.shape[0] * y.shape[1], 1))

            l = lossfn(X, y, w)
            print("Loss for this image is", l)
            sum_loss += l
            step += 1
            avg_loss = sum_loss / step

            # MLE Update
            w = w + lr * gradient_derivative(X, y, w)

        print("Average loss after Epoch: {} is {}".format(e + 1, avg_loss))
        # save the average loss, and the weights after every epoch of training
        state_saving(w, avg_loss, e + 1)

    avg_valofloss = 0
    avg_valofacc = 0
    avg_f1ofsc=0
    sum_loss1 = 0
    sumofacc = 0
    sumoff1 = 0
    step = 0

    # Validation metrics
    for file_name in os.listdir(val_img_path):
        X_Value = np.load(os.path.join(val_img_path, file_name))
        X_Value = np.reshape(X_Value, (X_Value.shape[0] * X_Value.shape[1], 3))
        o = np.ones((X_Value.shape[0], 1))
        X_Value = np.concatenate((o, X_Value), 1)
        y_Value = np.load(os.path.join(val_mask_path, file_name))
        y_Value = np.reshape(y_Value, (y_Value.shape[0] * y_Value.shape[1], 1))

        y_pred = np.matmul(X_Value, w) >= 0
        y_pred = y_pred.astype(np.uint8)

        # Metrics calculated here
        l = loss(X_Value, y_Value, w)
        step += 1
        sum_loss1 += l
        avg_valofloss = sum_loss1 / step
        acc = sklm.accuracy_score(y_Value, y_pred)
        sumofacc += acc
        avg_valofacc = sumofacc / step
        f1 = sklm.f1_score(y_Value, y_pred)
        sumoff1 += f1
        avg_f1ofsc = sumoff1 / step

        # Printing computed metrics from sklearn.metrics:
        print("Computed Loss for this image is:", l)
        print("Computed Accuracy for this image is:", acc)
        print("F1 metric for this image is: ", f1)

    # save the validation metrics over all the validation images to disk
    save_path = "./output/logistic_basic/"
    np.save(save_path + "val_loss" + ".npy", avg_valofloss)
    np.save(save_path + "val_acc" + ".npy", avg_valofacc)
    np.save(save_path + "val_f1" + ".npy", avg_f1ofsc)
    with open(save_path + "val_loss" + ".txt", "w") as fi:
        fi.write(str(l))
    with open(save_path + a_name + ".txt", "w") as fi:
        fi.write(str(a))
    with open(save_path + f_name + ".txt", "w") as fi:
        fi.write(str(f))

if __name__ == "__main__":
    #Siddarth A53299801
    main() #Calls the main function.
