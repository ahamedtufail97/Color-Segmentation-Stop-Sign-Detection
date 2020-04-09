#author @Siddarth
import os
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import cv2
import numpy as np

def main():
    twostops = [22, 38, 64, 68, 77, 91, 92, 95, 97]
    threestops = [93, 55]
    4achan = [51]
    4bchan = [150]
    done_a = [1, 2, 10, 11, 20, 21, 100]
    done_b = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 195, 196, 197, 198, 199, 200]
    # these images do not exist
    excludes = [7, 34, 35]
    excludes.extend(4achan)
    excludes.extend(done_a)
    4bchan.extend(done_b)

    # these are the images with STOP signs
    stops = [i for i in range(1, 101)]
    for i in excludes:
        stops.remove(i)
    # random permutation is good
    stops = np.random.permutation(stops)
    # Set aside 15 for validation set
    stop_val = stops[:15]
    # the rest are for training purposes
    stop_train = stops[15:]

    # those images that do NOT have stop signs
    other_ims = [i for i in range(101, 201)]
    for i in 4bchan:
        other_ims.remove(i)
    other_ims = np.random.permutation(other_ims)
    # set aside 15 images for validation
    other_val = other_ims[:15]
    # the rest of the images are for training purposes
    other_train = other_ims[15:]

    print(stop_train, stop_val)
    print(other_train, other_val)

    # ALL folder name initializations go here
    img_folder = "./trainset/"
    train_img_folder = "./data/train/images/"
    train_mask_folder = "./data/train/masks/"
    val_img_folder = "./data/val/images/"
    val_mask_folder = "./data/val/masks/"

    for file_name in os.listdir(img_folder):
        fn, ext = file_name.split(".")
        nu = int(fn)
        print("\n", nu)
        if nu in excludes or nu in 4bchan:
            continue
        img = cv2.imread(os.path.join(img_folder, file_name))
        print("Processing Image with name: ", file_name)
        print(type(img), img.dtype, img.shape, np.min(img), np.max(img))
        if img.dtype == 'uint8':
            # normalize uint8 to 0 to 1 float 32 for uniformity
            img = img.astype(np.float32) / 255.0
        else:
            print("Print this image is FINE!", img.dtype)

        disp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ### The 1st ROI
        # Image display
        fig = plt.figure()
        plt.imshow(disp_img)
        plt.title("Original Image")
        plt.show(block=False)

        # draw new ROI in green color
        my_roi1 = RoiPoly(color='g', fig=fig)

        # Show figure with ROI
        plt.imshow(disp_img)
        my_roi1.display_roi()
        plt.title("Displaying ROI!")
        plt.show(block=False)

        if nu in twostops:
            # Image display
            fig = plt.figure()
            plt.imshow(disp_img)
            plt.title("Original Image with 1 ROI")
            plt.show(block=False)

            # draw new ROI in blue color
            my_roi2 = RoiPoly(color='b', fig=fig)

            # Show figure with both ROI
            plt.imshow(disp_img)
            my_roi1.display_roi()
            my_roi2.display_roi()
            plt.title("Displaying 2 ROIs!")
            plt.show(block=False)

            # Extract mask
            img_gray = img[:, :, 0]
            rm1 = my_roi1.get_mask(img_gray)
            rm2 = my_roi2.get_mask(img_gray)
            red_mask = rm1 + rm2
            print(img_gray.dtype, img_gray.shape)
            print(red_mask.dtype, red_mask.shape, np.unique(red_mask))
            plt.imshow(red_mask, cmap=plt.cm.gray)
            plt.title('ROI mask of 2 ROIs')
            plt.show(block=False)
            red_mask = red_mask.astype(np.uint8)

        elif nu in threestops:
            # Image display
            fig = plt.figure()
            plt.imshow(disp_img)
            plt.title("Original Image with 1 ROI")
            plt.show(block=False)

            # draw new ROI in green color
            my_roi2 = RoiPoly(color='g', fig=fig)

            # Show figure with both ROI
            plt.imshow(disp_img)
            my_roi1.display_roi()
            my_roi2.display_roi()
            plt.title("Displaying 2 ROIs!")
            plt.show(block=False)

            # Image display
            fig = plt.figure()
            plt.imshow(disp_img)
            plt.title("Original Image with 2 ROIs")
            plt.show(block=False)

            # draw new ROI in yellow color
            my_roi3 = RoiPoly(color='y', fig=fig)

            # Show figure with all ROI
            plt.imshow(disp_img)
            my_roi1.display_roi()
            my_roi2.display_roi()
            my_roi3.display_roi()
            plt.title("Displaying all ROIs")
            plt.show(block=False)

            # Extract mask
            img_gray = img[:, :, 0]
            rm1 = my_roi1.get_mask(img_gray)
            rm2 = my_roi2.get_mask(img_gray)
            rm3 = my_roi3.get_mask(img_gray)
            red_mask = rm1 + rm2 + rm3
            print(img_gray.dtype, img_gray.shape)
            print(red_mask.dtype, red_mask.shape, np.unique(red_mask))
            plt.imshow(red_mask, cmap=plt.cm.gray)
            plt.title('ROI mask 3 ROIs')
            plt.show(block=False)
            red_mask = red_mask.astype(np.uint8)

        else:
            # Extract mask
            img_gray = img[:, :, 0]
            red_mask = my_roi1.get_mask(img_gray)
            print(img_gray.dtype, img_gray.shape)
            print(red_mask.dtype, red_mask.shape)
            plt.imshow(red_mask, cmap=plt.cm.gray)
            plt.title('ROI mask of my_roi')
            plt.show(block=False)
            red_mask = red_mask.astype(np.uint8)

        print("Final Sanity Check - ")
        assert(img.dtype == "float32")
        assert(img.shape[2] in [3, 4])
        assert(red_mask.dtype == "uint8")
        assert(np.min(red_mask) == 0)
        assert(np.max(red_mask) in [0, 1])
        print("All assertions passed! Saving RGB images now")

        # STOP train
        if nu in stop_train or nu in other_train:
            np.save(train_img_folder + fn + ".npy", img)
            np.save(train_mask_folder + fn + ".npy", red_mask)

        elif nu in stop_val or nu in other_val:
            np.save(val_img_folder + fn + ".npy", img)
            np.save(val_mask_folder + fn + ".npy", red_mask)

    print("Done Done and Done!\n\n")


if __name__ == "__main__":
    main()
