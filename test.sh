GPU_ID=2
CUDA_VISIBLE_DEVICES=$GPU_ID source Test/test.sh ../../data/ mod_adam_mse_0.00025_rgb_batch24_pretrainFalse_wlid0.1_wrgb0.1_wguide0.1_wpred1 ../../data/kitti_depth/val_selection_cropped/groundtruth_depth/
