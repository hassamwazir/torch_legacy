# torchrun --standalone --nnodes=1 --nproc_per_node=4  run.py --num_epochs=1 --mode=train --baselines=rt1/rt1x,octo/octo-base --batch_size=4 --checkpoint_dir=checkpoints4 --v=-2 --weight_decay=0 --lr=3e-5  

# torchrun --standalone --nnodes=1 --nproc_per_node=1  /scratch/whk240/llm/torch_legacy/run.py --project_name=rt1_bridge_oxe_75_nocheck_large_weight --oxe_datasets=ucsd_kitchen_dataset_converted_externally_to_rlds --num_epochs=5 --model=rt1 --batch_size=4 --checkpoint_dir=checkpoints4 --v=-2 --weight_decay=0 --lr=3e-5  

torchrun --standalone \
         --nnodes=1 \
         --nproc_per_node=1 \
         /scratch/whk240/llm/torch_legacy/run.py \
         --project_name=rt1_bridge_oxe_finetune \
         --oxe_datasets=ucsd_kitchen_dataset_converted_externally_to_rlds \
         --matmul_precision=medium \
         --num_epochs=5 \
         --model=rt1 \
         --batch_size=4 \
         --checkpoint_dir=checkpoints4 \
         --v=-2 \
         --weight_decay=0 \
         --lr=3e-5 \
         --lr_scheduler=cos \
         --future_action_window_size=5 \
         --strategy=ddp \
         --checkpoint_frequency=5000 \
         --num_parallel_calls 16 \
         --num_threads 16 \
         --shuffle_buffer_size 1000 \
         --gradient_clip_val 10000.0 \
         --seed 11 \
         --precision=16-mixed \
         --local_datasets /scratch/whk240/llm/torch_legacy/oxe_torch/episodes/pick_coke_can_place_left_of_spoon.hdf5 \
         --oxe_batch_percentage 0.75
