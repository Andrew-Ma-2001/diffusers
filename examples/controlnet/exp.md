bash
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/home/mayanze/PycharmProjects/diffusers/examples/controlnet/plane"

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




  --validation_image "/home/mayanze/PycharmProjects/diffusers/examples/controlnet/one_plane_label/P0005_cropped_keypoints.png"  \
 --validation_prompt "Airport area, runway and apron occupy a large part of the image, showing as dark and light concrete surfaces. Below, there are gray-black roads with several white and yellow small vehicles. On the left side, a wide-body large passenger plane is parked on the apron, predominantly white with yellow-gray patterns." \
 --validation_steps=100 \