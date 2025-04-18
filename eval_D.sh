#!/bin/sh

config=$1
gpus=$2
output=$3

if [ -z $config ]
then
    echo "No config file found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 3
opts=${@}




#ADE20k-150
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-vaihingen \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/Vaihingen.json" \
 DATASETS.TEST \(\"Vaihingen_all_sem_seg\"\,\) \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-potsdam \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/Potsdam.json" \
 DATASETS.TEST \(\"Potsdam_all_sem_seg\"\,\) \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts



 python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-isaid \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/iSAID.json" \
 DATASETS.TEST \(\"iSAID_all_sem_seg\"\,\) \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts




cat $output/eval-vaihingen/log.txt | grep copypaste
cat $output/eval-potsdam/log.txt | grep copypaste
cat $output/eval-isaid/log.txt | grep copypaste

