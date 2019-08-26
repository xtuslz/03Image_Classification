#!/usr/bin/env bash
python train_image_classifier.py \
    --train_dir=satellite/train_dir \
    --dataset_name=satellite \
    --dataset_split_name=train\
    --dataset_dir=satellite/data \
    --model_name=inception_v3 \
    --checkpoint_path=satellite/pretrained/inception_v3.ckpt \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --max_number_of_steps=100000 \
    --batch_size=8 \
    --learning_rate=0.001 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=300 \
    --save_summaries_secs=10 \
    --log_every_n_steps=1 \
    --optimizer=rmsprop \
    --weight_decay=0.00004
