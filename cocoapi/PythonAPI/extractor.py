import sys
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
#ABSOLUTE_DIR = '/home/derlee/coco/PythonAPI/'
TYPE = sys.argv[1]
sourceDir = TYPE + '2017'
saveDir = sys.argv[2]
category = sys.argv[3]
category_list = category.split(',')
print('Extracting {}/'.format(sourceDir))
print('Extracting class: {}'.format(category))
print('Saving to {}/{}'.format(saveDir, TYPE))
annFile = 'annotations/instances_{}.json'.format(sourceDir)
coco=COCO(annFile)
cats=coco.loadCats(coco.getCatIds())
#nms=[cat['name'] for cat in cats]
#print('COCO categories: \n{}\n'.format(' '.join(nms)))
#nms=set([cat['supercategory'] for cat in cats])
#print('COCO supercategories: \n{}\n'.format(' '.join(nms)))
catIds=coco.getCatIds(catNms=category_list)
imgIds=coco.getImgIds(catIds=catIds)
total_count = len(imgIds)
print('Total class images:{}'.format(total_count))
f = open(('{}/{}/{}_{}.txt'.format(saveDir, TYPE, TYPE, category)), 'w')
image_DIR = '{}/{}/image/'.format(saveDir, TYPE)
segmentation_DIR = '{}/{}/segmentation/'.format(saveDir, TYPE)
count = 0

for imgId in imgIds:
    img=coco.loadImgs(imgId)[0]
    #I = io.imread('image2017/{}/{}'.format(sourceDir, img['file_name']))
    #print(I.shape)
    #print(I.shape[:2])
    I = Image.open('image2017/{}/{}'.format(sourceDir, img['file_name']))
    #im = np.asarray(I)
    #print(im.shape)
    imgshape = tuple((I.size[1], I.size[0]))
    #print(imgshape)
    name = img['file_name'][:-3] + 'png'
    annIds=coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns=coco.loadAnns(annIds)
    image_name = '{}{}'.format(image_DIR, img['file_name'])
    segmentation_name = '{}{}'.format(segmentation_DIR, name)
    drew = coco.saveSegmentPxMap(anns, imgshape, segmentation_name)
    if drew > 0:
        rgbimg = Image.new("RGB", I.size)
        rgbimg.paste(I)
        rgbimg.save(image_name)
        #io.imsave(image_name, I)
        f.write('{} {}\n'.format(image_name, segmentation_name))
        #f.write('/' + (image_DIR + img['file_name'])+ ' ' + '/' +  segmentation_name + '\n')
        count += 1
        if count % 20 == 0:
            print("%d/%d  (%.2f %%)" % (count, total_count, 100*float(count)/total_count))
print('Extracted {}/{} images: {}/{} ({:.2f}%)'.format(TYPE, category, count, total_count, (100.0*count/total_count)))
print('Unable to draw: {}/{} ({:.2f}%)'.format((total_count - count), total_count, (100.0*(total_count - count)/total_count)))