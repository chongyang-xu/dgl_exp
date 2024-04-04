import os
import time

TOTAL_MACHINE=4 # totoal number of docker containers == total GPUs
N_EPOCH=500
N_EVAL=500 # evaluate every ...

def run_training_ctrl(dataset_list, num_part_list, use_first_n_parts, part_algo_list, num_layer_list, model_list, extra_tag="no"):
    part_algo_list = None
    for DATA_SET in dataset_list:
        for NUM_PART in num_part_list:
            for NUM_LAYER in num_layer_list:
                for MODEL in model_list:
                    first_n = NUM_PART if use_first_n_parts < 0 else use_first_n_parts
                    
                    n_mach = int(TOTAL_MACHINE) if first_n > TOTAL_MACHINE else int(first_n)
                    n_svr_per_mach = int(first_n / n_mach)
                    n_trainer_per_mach = int(n_svr_per_mach)
                    ip_config = f"docker{n_mach}"

                    CMD = f"python3 /workspace/dgl_dsg/experiments/tools/launch_train_ctrl.py" \
f" --graph_data_config /data/ds_pre/{DATA_SET}/part_n{NUM_PART}_combine_0_{first_n}.yaml" \
f" --use_first_n_parts {use_first_n_parts} --n_parts {NUM_PART}" \
f" --model {MODEL} --layers {NUM_LAYER} --one_id trainctrl_{DATA_SET}_{NUM_PART}_{first_n}_{extra_tag}_{MODEL}_{NUM_LAYER}_bdr" \
f" --batch_size 1000 --eval_every {N_EVAL} --n_epoch {N_EPOCH}" \
f" --n_mach {n_mach} --n_gpu_per_mach 1 --n_server_per_mach {n_svr_per_mach} --n_trainer_per_mach {n_trainer_per_mach} --n_sampler_per_trainer 0" \
f" --disable_backup_server True --ip_config {ip_config}" \
f" --sampling bdr" \
f" --log_path /workspace/dgl_dsg/experiments/logs/"
                    os.system(CMD)

def run_training(dataset_list, num_part_list, use_first_n_parts, part_algo_list, num_layer_list, model_list, extra_tag, num_halo, sampling_strategy):
    for DATA_SET in dataset_list:
        for NUM_PART in num_part_list:
            for PART_ALGO in part_algo_list:
                for NUM_LAYER in num_layer_list:
                    for MODEL in model_list:
                        first_n = use_first_n_parts if use_first_n_parts > 0 else NUM_PART
                    
                        n_mach = int(TOTAL_MACHINE) if first_n > TOTAL_MACHINE else int(first_n)
                        n_svr_per_mach = int(first_n / n_mach)
                        n_trainer_per_mach = int(n_svr_per_mach)
                        ip_config = f"docker{n_mach}"

                        CMD = f"python3 /workspace/dgl_dsg/experiments/tools/launch_train.py" \
f" --dataset {DATA_SET} --part_algo {PART_ALGO} --part_hop {num_halo} --n_parts {NUM_PART}" \
f" --use_first_n_parts {use_first_n_parts}" \
f" --model {MODEL} --layers {NUM_LAYER} --one_id train_{DATA_SET}_{NUM_PART}_{first_n}_{PART_ALGO}_{MODEL}_{NUM_LAYER}_{sampling_strategy}" \
f" --batch_size 1000 --eval_every {N_EVAL} --n_epoch {N_EPOCH}" \
f" --n_mach {n_mach} --n_gpu_per_mach 1 --n_server_per_mach {n_svr_per_mach} --n_trainer_per_mach {n_trainer_per_mach} --n_sampler_per_trainer 0" \
f" --disable_backup_server True --ip_config {ip_config}" \
f" --sampling {sampling_strategy}" \
f" --log_path /workspace/dgl_dsg/experiments/logs/"
#f" --part_conf /data/ds_pre/{DATA_SET}/data_part{NUM_PART}_{PART_ALGO}_{num_halo}_{first_n}/{DATA_SET}.json" \
                        os.system(CMD)
                        time.sleep(30)


#run_training(['ogbpr'], [4, 16], -1, ['random', 'metis'], [2, 3], ['gat'], extra_tag="test", num_halo=0, sampling_strategy="dft")
run_training(['ogbpa'], [4, 16], -1, ['random', 'metis', 'vcdeg', 'vcrandom', 'vcovlivious'], [2], ['gat'], extra_tag="test", num_halo=0, sampling_strategy="bdr")
run_training(['ogbpa'], [4, 16], -1, ['random', 'metis', 'vcdeg', 'vcrandom', 'vcovlivious'], [2, 3], ['gcn', 'sage'], extra_tag="test", num_halo=0, sampling_strategy="bdr")

#run_training_ctrl(['ogbpr'], [4], -1, None, [2], ['gcn'], extra_tag="test")

#run_training(['ogbpr', 'ogbpa'], [1, 4, 16, 64], -1, None, [2, 3, 4], ['gcn', 'sage', 'gat'], extra_tag="test", num_halo=0, sampling_strategy="bdr")

#run_training_ctrl(['ogbpr', 'ogbpa'], [1, 4, 16, 64], -1, None, [2, 3, 4], ['gcn', 'sage', 'gat'], extra_tag="test")

