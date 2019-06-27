import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw,ImageFilter
import PIL.ImageOps
import os
import sys


def sigmoid(x):
    return 1 / (1 + np.e ** -x)


activation_function = sigmoid
from scipy.stats import truncnorm


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)


class NeuralNetwork:
    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """
        A method to initialize the weight
        matrices of the neural network
        """
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0,
                             sd=1,
                             low=-rad,
                             upp=rad)
        self.wih = X.rvs((self.no_of_hidden_nodes,
                          self.no_of_in_nodes))
        # print(self.wih)
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.who = X.rvs((self.no_of_out_nodes,
                          self.no_of_hidden_nodes))

    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can
        be tuple, list or ndarray
        """

        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.wih,
                                input_vector)
        output_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.who,
                                output_hidden)
        output_network = activation_function(output_vector2)

        output_errors = target_vector - output_network
        # update the weights:
        tmp = output_errors * output_network \
              * (1.0 - output_network)
        tmp = self.learning_rate * np.dot(tmp,
                                          output_hidden.T)
        self.who += tmp
        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T,
                               output_errors)
        # update the weights:
        tmp = hidden_errors * output_hidden * \
              (1.0 - output_hidden)
        self.wih += self.learning_rate \
                    * np.dot(tmp, input_vector.T)

    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.wih, input_vector)

        output_vector = activation_function(output_vector)

        #print(len(self.who))
        #print(len(self.who[0]))
        output_vector = np.dot(self.who, output_vector)
        output_vector = activation_function(output_vector)
        return output_vector

    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm

    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()

    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs





def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    #print(tva)
    return tva






image_size = 28 # width and length

import cv2
r=0
value=sys.argv[1]
print(value)
img = cv2.imread(value, 0)
#im = Image.open("1.png")   
cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
counter=0
contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
arr=[]


for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    
    # draw a white rectangle to visualize the bounding rect
    cv2.rectangle(img, (x, y), (x + w, y + h), 0, 1)

    width, height = x,y
    roi = img[y-10:y+h+20, x-10:x+w+20]

    

    
    str1=str(counter)
    str3='new'
    #str2='C:\Users\sidhartha\Desktop\mnist'
    name=str1+'.png'
    #name2=str3+'.png'
    cv2.imwrite(name, roi)

    
    image = Image.open(name)
    inverted_image = PIL.ImageOps.invert(image)
    inverted_image.save(name)
    x=[imageprepare(name)]
    newArr=[[0 for d in range(28)] for y in range(28)]
    k = 0
    for i in range(28):
        for j in range(28):
            newArr[i][j]=x[0][k]
            k=k+1

    k=1
    val=[0]*785
    
    for i in range(28):
        for j in range(28):
            val[k]=newArr[i][j]
            k=k+1

    import csv

    csvData = val

    with open('test2.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(csvData)

    csvFile.close()
    image_size = 28  # width and length
    no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    data_path = ""
    train_data = np.loadtxt(data_path + "mnist_train.csv",
                        delimiter=",")
    test_data = np.loadtxt(data_path + "test2.csv",
                       delimiter=",")
    #print(train_data[0])

    fac = 255 * 0.99 + 0.01
    train_imgs = np.asfarray(train_data[:, 1:]) / fac
    test_imgs = np.asfarray(test_data[1:])
    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:1])

    lr = np.arange(10)
    lr = np.arange(no_of_different_labels)
    # transform labels into one hot representation
    train_labels_one_hot = (lr == train_labels).astype(np.float)
    test_labels_one_hot = (lr == test_labels).astype(np.float)
    # we don't want zeroes and ones in the labels neither:
    train_labels_one_hot[train_labels_one_hot == 0] = 0.01
    train_labels_one_hot[train_labels_one_hot == 1] = 0.99
    test_labels_one_hot[test_labels_one_hot == 0] = 0.01
    test_labels_one_hot[test_labels_one_hot == 1] = 0.99

    ANN = NeuralNetwork(no_of_in_nodes=image_pixels,
                    no_of_out_nodes=10,
                    no_of_hidden_nodes=100,
                    learning_rate=0.01)

    for i in range(len(train_imgs)):
        ANN.train(train_imgs[i], train_labels_one_hot[i])


    
    res = ANN.run(test_imgs)
    print(test_labels, np.argmax(res), np.max(res))
    print(res)
    res = ANN.run(test_imgs)
    res_max = res.argmax()
    res_max=res_max.item()
    arr.append(res_max)
   
    
    counter=counter+1
print("predicted value:")
print(arr)
    



cv2.drawContours(img, contours, -1, (0, 0, 0), 2)

cv2.imwrite("output.png",img)


