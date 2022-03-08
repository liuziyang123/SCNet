GPU_ID=0,1,2,3
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --data_path ../../data/ --learning_rate 0.00025  --weight_decay 0 --lr_policy plateau --pretrained false --batch_size 24 --multi --gpu_ids 0,1,2,3 --no_tb False --print_freq 300 #--directly_path ../pretrained/model_best_epoch.pth.tar

