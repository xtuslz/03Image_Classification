python eval_image_classifier.py \
    --checkpoint_path=satellite/train_dir \
    --eval_dir=satellite/eval_dir \
    --dataset_name=satellite \
    --dataset_split_name=validation \
    --dataset_dir=satellite/data \
    --model_name=inception_v3
