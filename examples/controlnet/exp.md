bash
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export MODEL_DIR="/home/mayanze/PycharmProjects/sdxl_demo1/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/home/mayanze/PycharmProjects/diffusers/examples/controlnet/plane"
export OUTPUT_DIR='/home/mayanze/PycharmProjects/diffusers/examples/controlnet/boat'
export OUTPUT_DIR='/home/mayanze/PycharmProjects/diffusers/examples/controlnet/ocean'
export OUTPUT_DIR='/home/mayanze/PycharmProjects/diffusers/examples/controlnet/mar20'
export CUDA_VISIBLE_DEVICES=4,5,6,7

accelerate launch train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/home/mayanze/PycharmProjects/diffusers/examples/controlnet/one_plane_label/one_plane_dataset.csv \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --validation_image "/home/mayanze/PycharmProjects/diffusers/examples/controlnet/one_plane_label/P0005_cropped_keypoints.png"  \
 --validation_prompt "Airport area, runway and apron occupy a large part of the image, showing as dark and light concrete surfaces. Below, there are gray-black roads with several white and yellow small vehicles. In the lower right corner and along the road edge, there is brown-green vegetation. On the left side, a wide-body large passenger plane is parked on the apron, predominantly white with yellow-gray patterns." \
 --validation_steps=100 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42


accelerate launch train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/home/mayanze/PycharmProjects/diffusers/examples/controlnet/one_plane_label/one_plane_dataset.csv \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --validation_image "/home/mayanze/PycharmProjects/diffusers/examples/controlnet/one_plane_label/P0005_cropped_keypoints.png"  \
 --validation_prompt "Airport area, runway and apron occupy a large part of the image, showing as dark and light concrete surfaces. Below, there are gray-black roads with several white and yellow small vehicles. In the lower right corner and along the road edge, there is brown-green vegetation. On the left side, a wide-body large passenger plane is parked on the apron, predominantly white with yellow-gray patterns." \
 --validation_steps=100 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42



accelerate launch train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/home/mayanze/PycharmProjects/diffusers/examples/controlnet/one_plane_label/one_plane_dataset.csv \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --validation_image "/home/mayanze/PycharmProjects/diffusers/examples/controlnet/one_plane_label/P0005_cropped_keypoints.png"  \
 --validation_prompt "Airport area, runway and apron occupy a large part of the image, showing as dark and light concrete surfaces. Below, there are gray-black roads with several white and yellow small vehicles. In the lower right corner and along the road edge, there is brown-green vegetation. On the left side, a wide-body large passenger plane is parked on the apron, predominantly white with yellow-gray patterns." \
 --validation_steps=100 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42


python train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/home/mayanze/PycharmProjects/diffusers/examples/controlnet/one_plane_label/one_plane_dataset.csv \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --gradient_checkpointing \
 --use_8bit_adam \
 --seed=42

python train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/home/mayanze/PycharmProjects/diffusers/examples/controlnet/ResizedImage/hrsc_filtered_data.csv \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --gradient_checkpointing \
 --use_8bit_adam \
 --seed=42

python train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/home/mayanze/PycharmProjects/diffusers/examples/controlnet/ocean/ocean.csv \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --gradient_checkpointing \
 --use_8bit_adam \
 --seed=42


python train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/home/mayanze/PycharmProjects/diffusers/examples/controlnet/AllPlaneKeypoint/box_plane.csv \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --seed=42




  --validation_image "/home/mayanze/PycharmProjects/diffusers/examples/controlnet/one_plane_label/P0005_cropped_keypoints.png"  \
 --validation_prompt "Airport area, runway and apron occupy a large part of the image, showing as dark and light concrete surfaces. Below, there are gray-black roads with several white and yellow small vehicles. On the left side, a wide-body large passenger plane is parked on the apron, predominantly white with yellow-gray patterns." \
 --validation_steps=100


export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR='/home/mayanze/PycharmProjects/diffusers/output/plane-sd'
export CUDA_VISIBLE_DEVICES=2,3,4,5

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/home/mayanze/PycharmProjects/diffusers/examples/controlnet/AllPlaneKeypoint/box_plane.csv \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=20000 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4

在 train_controlnet 里面修改了文件读取路径，变成本地 csv 的形式，sd 1.5 是可以多卡的，但是 sdxl 不可以；两个模型精度不一样，sdxl fp8， sd 1.5 应该是 fp16


export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OUTPUT_DIR='/home/mayanze/PycharmProjects/diffusers/output/plane-whole-sdxl'
export DATASET_NAME='/home/mayanze/PycharmProjects/diffusers/examples/controlnet/PlaneKeypointCOCO_whole_v2/whole_box_plane.csv'

python train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=$DATASET_NAME \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=20000 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --seed=42


cd examples/controlnet/
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR='/home/mayanze/PycharmProjects/diffusers/output/plane-whole'
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch --main_process_port=25901 train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/home/mayanze/PycharmProjects/diffusers/examples/controlnet/PlaneKeypointCOCO_whole_v2/whole_box_plane.csv \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=20000 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4


export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OUTPUT_DIR='/home/mayanze/PycharmProjects/diffusers/output/plane-whole-segskeleton'
export DATASET_NAME='/home/mayanze/PycharmProjects/diffusers/examples/controlnet/image_pairs.csv'

python train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=$DATASET_NAME \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=20000 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --seed=42


export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export OUTPUT_DIR='/home/mayanze/PycharmProjects/diffusers/output/plane-whole-MYsegskeleton'
export DATASET_NAME='/home/mayanze/PycharmProjects/diffusers/examples/controlnet/image_pairs_MY.csv'

python train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=$DATASET_NAME \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=20000 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --seed=42 \
 --resume_from_checkpoint "latest"

