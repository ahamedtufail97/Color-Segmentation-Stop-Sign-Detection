"""
ECE276A WI20 HW1
Stop Sign Detector
"""

import cv2
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches


class StopSignDetector():
    def __init__(self):
        """
            Initialize your stop sign detector with the attributes you need,
            e.g., parameters of your classifier
        """
        self.load_path = "final_weights.npy" #Using the weights I got by logistic regression
        self.w = self.load_weights()

    def load_weights(self):
        """
        Load the weights trained by my model

        Inputs:
            pth - the path to load the weights from

        Outputs:
            w - the weights trained by my model
        """
        w = np.load(self.load_path)
        return w

    def segment_image(self, img):
        """
            Obtain a segmented image using a color classifier,
            e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture,
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        """
        # YOUR CODE HERE
        # author @Siddarth A53299801
        # normalize image
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = img / 255.0

        sh = img.shape
        X = np.reshape(img, (sh[0] * sh[1], 3))
        o = np.ones((X.shape[0], 1))
        X = np.concatenate((o, X), 1)

        # Load the weights
        w = self.w

        print("\nBounding Box - B.B - Statistics are as follows: ")
        print(X.shape, w.shape)
        print(X.dtype, np.min(X), np.max(X))

        # predicting the mask
        y_pred = np.matmul(X, w) >= 0
        y_pred = y_pred.astype(np.uint8)
        print("YPred Stats: ", y_pred.shape, y_pred.dtype, np.min(y_pred), np.max(y_pred))
        mask_img = np.reshape(y_pred, (sh[0], sh[1]))
        print("Mask Image Stats: ", mask_img.shape, mask_img.dtype, np.min(mask_img), np.max(mask_img))
        return mask_img

    def get_bounding_box(self, img):
        """
            Find the bounding box of the stop sign
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                is from left to right in the image.

            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        """
        # YOUR CODE HERE
        #author @Siddarth
        mask_img = self.segment_image(img)

        # dilate mask_img
        kernel = np.ones((3, 3), np.uint8) # Using 3x3 Kernel size for dilation
        dilated = cv2.dilate(mask_img, kernel, iterations=2)

        #cntrs, h = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntrs, h = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h = h[0]
        #print("CONTOURS:", len(cntrs), cntrs)
        #print("Heirarchies:", len(h), h)

        # Polygon Processing
        num_sides = [i for i in range(5, 19)]

        boxes = []
        max_h = mask_img.shape[0]
        max_w = mask_img.shape[1]
        img_area = max_h*max_w

        for c in zip(cntrs, h):
            cur_c = c[0]
            cur_h = c[1]


            thresh = 0.01 * cv2.arcLength(cur_c, True)
            poly_aprox = cv2.approxPolyDP(cur_c, thresh, True)

            # top left, width, height
            x1, y1, w, h = cv2.boundingRect(cur_c)
            ecc = w / h
            are = w * h
            if ecc < 1.2 and ecc > 0.8:
                if are / img_area > 0.008: 
                    print("Contour for this image has ", len(poly_aprox), "sides")
                    boxes.append([x1, max_h - (y1 + h), x1+w, max_h - y1])
        # sort left to right
        # boxes.sort(key=lambda x: x[0])
        print(len(boxes), boxes)
        return boxes


if __name__ == '__main__':
    folder = "trainset"
    my_detector = StopSignDetector()
    for filename in os.listdir(folder):
        # read one test image
        #filename = "38.jpg"
        img = cv2.imread(os.path.join(folder, filename))
        print(img.shape, img.dtype, np.min(img), np.max(img))
        #cv2.imshow('image', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Display Segmented images:

        mask_img = my_detector.segment_image(img)
        '''#Uncomment to display figure
        fig = plt.figure()
        plt.imshow(mask_img, cmap=plt.cm.gray)
        plt.title("Segmentation Mask")
        plt.show()
        '''
        # dilate mask_img
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask_img, kernel, iterations=2)

        #fig = plt.figure()
        #plt.imshow(dilated, cmap=plt.cm.gray)
        #plt.title("Dilated Segmentation Mask") 
        #plt.show()

        # Display Stop sign bounding box
        boxes = my_detector.get_bounding_box(img)
        print(boxes)

        # plot the boxes on the original image
        im_h, im_w = mask_img.shape
        disp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_area = im_w * im_h
        fig, ax = plt.subplots(1)
        ax.imshow(disp_img)

        # plot the rectangles now
        for coord in boxes:
            x_bottom, y_bottom, x_top, y_top = coord
            w = x_top - x_bottom #width
            h = y_top - y_bottom #height
            x = x_bottom
            y = im_h - y_top
            #print(x,y,w,h)
            bb_area = w * h
            ratio = bb_area / img_area
            wh_ratio = h / w
            print(wh_ratio)
            print(ratio)
            rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
    plt.show()
        # The autograder checks your answers to the functions segment_image() and get_bounding_box()
        # Make sure your code runs as expected on the testset before submitting to Gradescope
