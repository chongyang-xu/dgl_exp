import os
import time

TOTAL_MACHINE=4 # totoal number of docker containers == total GPUs
N_EPOCH=40
N_EVAL=200 # evaluate every ...
BATCH_SIZE=1
EVAL_BATCH_SIZE=1000 # ogbpax16 use 1e3, ogbpr used 1e5

for i in range(0,4):
    os.system(f"ssh ds4gnn_w{i} \"pkill -9 python3; ps -aux| grep python\"")
for i in range(0,4):
    os.system(f"ssh ds4gnn_w{i} \"pkill -9 python3; ps -aux| grep python\"")

def run_training_ctrl(dataset_list, num_part_list, use_first_n_parts, part_algo_list, num_layer_list, model_list, extra_tag="no", switch_mode=4):
    part_algo_list = None
    for DATA_SET in dataset_list:
        for NUM_PART in num_part_list:
            for NUM_LAYER in num_layer_list:
                for MODEL in model_list:

                    first_n = NUM_PART if use_first_n_parts < 0 else use_first_n_parts
                    
                    n_mach = int(TOTAL_MACHINE) if first_n > TOTAL_MACHINE else int(first_n)
                    n_svr_per_mach = int(first_n / n_mach)
                    n_trainer_per_mach = int(n_svr_per_mach) 
                    if NUM_PART == 256:
                        n_trainer_per_mach = int(n_trainer_per_mach/2)
                    print(f"N_trainer: {n_trainer_per_mach}")
                    time.sleep(3)

                    ip_config = f"docker_a_{n_mach}" 

                    CMD = f"python3 /workspace/dgl_dsg/experiments/tools/launch_train_from_chunks.py" \
f" --dataset {DATA_SET} --use_first_n_parts {use_first_n_parts} --n_parts {NUM_PART}" \
f" --model {MODEL} --layers {NUM_LAYER} --one_id dbg_trainctrl_{DATA_SET}_{NUM_PART}_{first_n}_{extra_tag}_{MODEL}_{NUM_LAYER}_bdr" \
f" --batch_size {BATCH_SIZE} --batch_size_eval {EVAL_BATCH_SIZE} --eval_every {N_EVAL} --n_epoch {N_EPOCH}" \
f" --n_mach {n_mach} --n_gpu_per_mach 1 --n_server_per_mach {n_svr_per_mach} --n_trainer_per_mach {n_trainer_per_mach} --n_sampler_per_trainer 0" \
f" --disable_backup_server True --ip_config {ip_config}" \
f" --sampling bdr" \
f" --log_path /workspace/dgl_dsg/experiments/logs/"
                    os.system(CMD)
#                    CMD = f"python3 /workspace/dgl_dsg/experiments/tools/launch_train_from_chunks.py" \
#f" --graph_data_config /data/ds_pre/{DATA_SET}/part_n{NUM_PART}_combine_0_{first_n}.yaml" \
#f" --use_first_n_parts {use_first_n_parts} --n_parts {NUM_PART}" \
#f" --model {MODEL} --layers {NUM_LAYER} --one_id trainctrl_{DATA_SET}_{NUM_PART}_{first_n}_{extra_tag}_{MODEL}_{NUM_LAYER}_bdr" \
#f" --batch_size 1000 --batch_size_eval {EVAL_BATCH_SIZE} --eval_every {N_EVAL} --n_epoch {N_EPOCH}" \
#f" --n_mach {n_mach} --n_gpu_per_mach 1 --n_server_per_mach {n_svr_per_mach} --n_trainer_per_mach {n_trainer_per_mach} --n_sampler_per_trainer 0" \
#f" --disable_backup_server True --ip_config {ip_config}" \
#f" --train_ctrl_mode {switch_mode}" \
#f" --sampling bdr" \
#f" --log_path /workspace/dgl_dsg/experiments/logs/"
#                    os.system(CMD)
                    for i in range(0, 4):
                        os.system(f"ssh ds4gnn_w{i} \"pkill -9 python3; ps -aux| grep python\"")
                    for i in range(0, 4):
                        os.system(f"ssh ds4gnn_w{i} \"pkill -9 python3; ps -aux| grep python\"")

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
                        if NUM_PART == 256:
                            n_trainer_per_mach = int(n_trainer_per_mach/2)
                        print(f"N_trainer: {n_trainer_per_mach}")
                        ip_config = f"docker_a_{n_mach}"

                        CMD = f"python3 /workspace/dgl_dsg/experiments/tools/launch_train.py" \
f" --dataset {DATA_SET} --part_algo {PART_ALGO} --part_hop {num_halo} --n_parts {NUM_PART}" \
f" --use_first_n_parts {use_first_n_parts}" \
f" --model {MODEL} --layers {NUM_LAYER} --one_id dbg_train_{DATA_SET}_{NUM_PART}_{first_n}_{PART_ALGO}_{MODEL}_{NUM_LAYER}_{sampling_strategy}" \
f" --batch_size {BATCH_SIZE} --batch_size_eval {EVAL_BATCH_SIZE} --eval_every {N_EVAL} --n_epoch {N_EPOCH}" \
f" --n_mach {n_mach} --n_gpu_per_mach 1 --n_server_per_mach {n_svr_per_mach} --n_trainer_per_mach {n_trainer_per_mach} --n_sampler_per_trainer 0" \
f" --disable_backup_server True --ip_config {ip_config}" \
f" --sampling {sampling_strategy}" \
f" --log_path /workspace/dgl_dsg/experiments/logs/"
#f" --part_conf /data/ds_pre/{DATA_SET}/data_part{NUM_PART}_{PART_ALGO}_{num_halo}_{first_n}/{DATA_SET}.json" \
                        os.system(CMD)
                        for i in range(0, 4):
                            os.system(f"ssh ds4gnn_w{i} \"pkill -9 python3; ps -aux| grep python\"")
                        for i in range(0, 4):
                            os.system(f"ssh ds4gnn_w{i} \"pkill -9 python3; ps -aux| grep python\"")
"""
for dataset in ['ogbpr']:
    for i in range(3):
        extra_tag = "withselfloop_xno_part_bias_500x"
        run_training_ctrl([dataset], [16], -1, None, [2], ['gcn'], extra_tag=extra_tag, switch_mode=3)
        run_training_ctrl([dataset], [16], -1, None, [2], ['sage'], extra_tag=extra_tag, switch_mode=3)
        run_training_ctrl([dataset], [16], -1, None, [2], ['gat'], extra_tag=extra_tag, switch_mode=3)
        run_training_ctrl([dataset], [16], -1, None, [3], ['gcn'], extra_tag=extra_tag, switch_mode=3)
        run_training_ctrl([dataset], [16], -1, None, [3], ['sage'], extra_tag=extra_tag, switch_mode=3)
        run_training_ctrl([dataset], [16], -1, None, [3], ['gat'], extra_tag=extra_tag, switch_mode=3)
"""

#run_training(['reddit', 'ogbpr', 'cora', 'ogbar'], [16], -1, ['metis'], [4], ['gcn'], extra_tag="default", num_halo=1, sampling_strategy="dft")
#run_training(['cora', 'ogbar'], [16], -1, ['metis'], [4], ['sage'], extra_tag="default", num_halo=1, sampling_strategy="dft")
# reddit, ogbpr gat oom
#run_training(['cora', 'ogbar'], [16], -1, ['metis'], [4], ['gat'], extra_tag="default", num_halo=1, sampling_strategy="dft")
#run_training([ 'ogbpr' ], [16], -1, ['metis'], [2], ['sage'], extra_tag="default", num_halo=1, sampling_strategy="dft")
#extra_tag="useless"
#run_training_ctrl(['ogbpr'], [16], -1, None, [3], ['sage'], extra_tag=extra_tag, switch_mode=3)
#run_training_ctrl(['reddit'], [16], -1, None, [2], ['sage'], extra_tag=extra_tag, switch_mode=3)
#run_training_ctrl(['reddit'], [16], -1, None, [3], ['sage'], extra_tag=extra_tag, switch_mode=3)
#exit(0)

#run_training([ 'reddit' ], [16], -1, ['metis'], [2], ['sage'], extra_tag="default", num_halo=1, sampling_strategy="dft")

run_training([ 'ogbpr' ], [16], -1, ['metis'], [3], ['sage'], extra_tag="default", num_halo=1, sampling_strategy="dft")
run_training([ 'reddit' ], [16], -1, ['metis'], [3], ['sage'], extra_tag="default", num_halo=1, sampling_strategy="dft")

exit(0)
extra_tag="useless"
run_training_ctrl(['ogbpr'], [16], -1, None, [3], ['sage'], extra_tag=extra_tag, switch_mode=3)

extra_tag="useless"
run_training_ctrl(['cora'], [16], -1, None, [2], ['sage'], extra_tag=extra_tag, switch_mode=3)
exit(0)
run_training([ 'cora' ], [16], -1, ['metis'], [2], ['sage'], extra_tag="default", num_halo=1, sampling_strategy="dft")
exit(0)

run_training([ 'ogbpr' ], [16], -1, ['metis'], [2], ['sage'], extra_tag="default", num_halo=1, sampling_strategy="dft")

# sample 20, then switch, sample another 20
exit(0)
extra_tag="useless"
run_training_ctrl(['ogbpr'], [16], -1, None, [2], ['sage'], extra_tag=extra_tag, switch_mode=3)

run_training([ 'cora', 'ogbar', 'reddit'], [16], -1, ['metis'], [2], ['sage'], extra_tag="default", num_halo=1, sampling_strategy="dft")
run_training_ctrl(['cora', 'ogbar', 'reddit'], [16], -1, None, [2], ['sage'], extra_tag=extra_tag, switch_mode=3)

exit(0)
run_training([ 'ogbpr'], [16], -1, ['metis'], [4], ['sage'], extra_tag="default", num_halo=1, sampling_strategy="dft")
