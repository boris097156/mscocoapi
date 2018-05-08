#! /bin/bash
TYPE=$1 #train or val
SAVEDIR=$2 #dir to save to
CATEGORY=$3 #which category to extract
mkdir $SAVEDIR/
mkdir $SAVEDIR/$TYPE
mkdir $SAVEDIR/$TYPE/image
mkdir $SAVEDIR/$TYPE/segmentation
python cocoapi/PythonAPI/extractor.py $TYPE $SAVEDIR $CATEGORY
