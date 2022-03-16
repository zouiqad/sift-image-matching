import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

path_query = 'queries'
path_train = 'License Plates'

query_imgs = []
train_imgs = []
classNames = []
myList = os.listdir(path_query)

print('nb of classes', len(myList))

for cl in myList:
    img = cv2.imread(f'{path_query}/{cl}', 0)
    query_imgs.append(img)
    classNames.append(os.path.splitext(cl)[0])

myList = os.listdir(path_train)
for tr in myList:
    img = cv2.imread(f'{path_train}/{tr}', 0)
    train_imgs.append(img)

print(classNames)
print('nb of train imgs:', len(train_imgs))


def match(train, queries):
    results = []
    sift = cv2.SIFT_create()

    for imgQ in queries:

        kp2, des2 = sift.detectAndCompute(imgQ, None) #query img
        loca_maximum = []

        for imgT in train:
            kp1, des1 = sift.detectAndCompute(imgT, None) #train img
            #Feature matching
            #Create BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

            #Match descriptors
            matches = bf.knnMatch(des1, des2, k=2)

            # ratio test (store all good matches) 
            good = []
            for m,n in matches:
                if m.distance < 0.70*n.distance:
                    good.append([m])
            
            if len(loca_maximum)<len(good):
                loca_maximum = list(good)
                best_match = imgT, kp1
        

        #match the query to the train
        results.append([best_match[0], best_match[1], imgQ, kp2, loca_maximum])
        
    return results

results = match(train_imgs, query_imgs)
    

for res in results:
        img3 = cv2.drawMatchesKnn(res[0], res[1], res[2], res[3], res[4], None, flags=2)
        plt.imshow(img3)
        plt.show()
