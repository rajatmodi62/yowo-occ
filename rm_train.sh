#train jhmdb 

# CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfg/jhmdb_combined_train.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfg/jhmdb_iid_train.yaml
# CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfg/jhmdb_ood_train.yaml
