
####################### ucf #################################


CUDA_VISIBLE_DEVICES=1 python rm_test_main.py --cfg cfg/ucf24.yaml
CUDA_VISIBLE_DEVICES=1 python rm_test_main.py --cfg cfg/jhmdb.yaml

# clean ucf 

#train code 
# python main.py --cfg cfg/ucf24.yaml

# #eval code 
# python evaluation_ucf24_jhmdb/pascalvoc.py --gtfolder /media/yogesh/rm2/occlusion_work/YOWO/evaluation_ucf24_jhmdb/groundtruths_ucf --detfolder /media/yogesh/rm2/occlusion_work/YOWO/ucf_clean/detections_0/


# ############## jhmdb 


# #clean jhmdb 
# python main.py --cfg cfg/jhmdb.yaml

# #eval 
# python video_mAP.py --cfg cfg/train_configs/jhmdb.yaml --dataset jhmdb21
