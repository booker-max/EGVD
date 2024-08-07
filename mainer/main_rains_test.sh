python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port 29501 /code/EGVD/mainer/main_new.py \
        --save_dir /data/booker/LN_base/TMM/exp/EventRepresentation/event_image \
        --data_type NN \
        --test_data_type N_N \
        --train_dataset_type ELNRainDataset \
        --train_dataset SequenceDataset \
        --test_dataset SequenceDataset_test \
        --trainer_type trainer_gtanet \
        --baseline_type base_ours \
        --ablation_type MSEG_e \
        --main_mode test \
        --enable_vis \
        --base_path /code/EGVD/options/base.yaml \
        --sequence_length 7 \
        --test_sequence_length 3 \
        --batch_size 4 \
        --batch_size_val 4 \
        --batch_size_test 1 \
        --epochs 500