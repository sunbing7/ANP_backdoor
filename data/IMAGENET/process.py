from scipy import misc
import os
import numpy as np
import pandas as pd
'''
images=os.listdir('./images/')
k=1
for image in images:
    os.rename('./images/'+image,'./images/'+'{0}.png'.format(k))
    k=k+1
'''

# f = open('./labels.txt', 'w+')
# df = pd.read_csv('dev_dataset.csv',usecols=[0, 6]) # read ImageId and TrueLabel
# imagesid=df['ImageId'][:]
# labels=df['TrueLabel'][:]
# x=zip(imagesid,labels)
# x=sorted(x)
# for i in range(1000):
#     f.write(str(x[i][1])+'\n')
# f.close()
'''
images=os.listdir('./images/')
data = pd.read_csv("dev_dataset.csv")
classes = data['TrueLabel'].tolist()
for class_ in classes:
    dir = './images/' + str(class_)
    if os.path.exists(dir):
        continue
    else:
        os.mkdir(dir)

for i in range (1, 1001):
    os.rename('./images/' + str(i) + '.png', './images/' + str(classes[i - 1]) + '/' + str(i) + '.png')
'''
'''
folders = list(os.walk('./images/'))[1:]
for folder in folders:
    # folder example: ('FOLDER/3', [], ['file'])
    if not folder[2]:
        os.rmdir(folder[0])
'''

#images=os.listdir('./all/')
images = sorted( filter( lambda x: os.path.isfile(os.path.join('./all/', x)),
                        os.listdir('./all/') ) )
k = 0
for image in images:
    dir = './images/' + str(k).zfill(3)
    if not os.path.exists(dir):
        os.mkdir(dir)

    os.rename('./all/' + image, dir + '/' + image)
    k = k + 1

