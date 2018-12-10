
import numpy as np
import cv2
import glob
import os


def pedestrian_loader():
    image_list = []
    path = os.path.abspath("PennFudanPed/PNGImages/*.png")
    for filename in glob.glob(path):
        im = cv2.imread(filename, 0)
        image_list.append(im)
    return np.asarray(image_list)


def pedestrian_box_border_information():
    train_labels = []
    path = os.path.abspath("PennFudanPed/Annotation/*.txt")
    for filename in glob.glob(path):
        with open(filename) as file:
            boundary_data = []
            for line in file:
                if "Bounding box for object " in line:
                    # View Annotations for more info
                    theInfo = line.split(":")[1].replace("-", "").replace(" ", "").replace("(", "").replace(")", ",")[:-2].split(",")
                    for i in theInfo:
                        boundary_data.append(int(i))

                    # counter = 0
                    # toAdd = []
                    # toAppend = []
                    # for i in theInfo:
                    #     if counter == 0:
                    #         toAppend.append(int(i))
                    #         counter +=1
                    #     elif counter == 1:
                    #         toAppend.append(int(i))
                    #         toAdd.append(toAppend)
                    #         counter = 0
                    #         toAppend = []
                    #     if len(toAdd) == 2:
                    #         boundary_data.append(toAdd)
                    #         toAdd = []
                    #         toAppend = []
            train_labels.append(np.asarray(boundary_data))
    return train_labels





if __name__ =="__main__":
    pedestrian_box_border_information()