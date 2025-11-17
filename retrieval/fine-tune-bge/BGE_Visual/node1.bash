GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR= # your master address
MASTER_PORT= # your master port
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/home/sonlt/drive/fact_check_end2end/data/train_image_candidates.jsonl
SAVE_PATH=/home/sonlt/drive/fact_check_end2end/model/BGE-base-Visualized-Mocheg
IMAGE_PATH=/home/sonlt/drive/data/mocheg/images
EPOCH=8
# RESUME_PATH=BAAI/bge-visualized-m3
SAVE_STEPS=100000
GROUP_SIZE=6 # = one (positive sample) + number (of hard negative samples)
BSZ_PERGPU=16
LR=1e-5

# Training_Dir= #your training dir
DeepSpeedConfig=/home/sonlt/drive/fact_check_end2end/ds_stage0.json
# cd $Training_Dir
# Data and model


mkdir $SAVE_PATH
# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE"

export CUDA_VISIBLE_DEVICES=0
export LAUNCHER="python -m torch.distributed.run \
    $DISTRIBUTED_ARGS \
"

full_options="
  --output_dir $SAVE_PATH \
  --bge_model_name_or_path BAAI/bge-base-en-v1.5 \
  --visual_model_name_or_path EVA02-CLIP-B-16 \
  --dataloader_num_workers 1  \
  --train_data $DATA_PATH \
  --train_data_image $IMAGE_PATH \
  --train_group_size $GROUP_SIZE
  --learning_rate $LR \
  --fp16 \
  --per_device_train_batch_size $BSZ_PERGPU \
  --dataloader_drop_last True \
  --normlized True \
  --temperature 0.02 \
  --logging_steps 1000 \
  --num_train_epochs $EPOCH \
  --negatives_cross_device \
  --train_text_tower True  \
  --train_vision_tower True \
  --save_steps $SAVE_STEPS \
  "

run_cmd="$LAUNCHER -m run_ds_cirr ${full_options}"
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee $SAVE_PATH/output_$NODE_RANK.log



set +x

