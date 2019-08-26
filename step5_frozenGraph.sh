python freeze_graph.py \
--input_graph slim/satellite/inception_v3_inf_graph.pb \
--input_checkpoint slim/satellite/train_dir/model.ckpt-454 \
--input_binary true \
--output_node_names InceptionV3/Predictions/Reshape_1 \
--output_graph slim/satellite/frozen_graph.pb
