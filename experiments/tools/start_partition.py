import os
import time

USED_ALGO_IN_COMBINE = ['random', 'metis', 'vcrandom', 'vcoblivious', 'vcdeg']
def combine_yaml_head(dataset, num_part):
    combine_yaml = "description: \"this yaml file specifies a group of partitions of a graph. for each partition, the unique name and part_conf is required\""
    combine_yaml += "\n"
    combine_yaml += f"dataset: \"{dataset}\""
    combine_yaml += "\n"
    combine_yaml += f"num_parts: {num_part}"
    combine_yaml += "\n"

    if dataset == "ogbpr":
        n_class = 47
    elif dataset == "ogbpa":
        n_class = 172
    else:
        assert False
    combine_yaml += f"n_classes: {n_class}"
    combine_yaml += "\n"
    combine_yaml += "partitions:"
    combine_yaml += "\n"
    return combine_yaml


def run(dataset_list, num_parts_list, save_first_n, halo_hop=0):
    DATA_ROOT="/data"
    for DATA_SET in dataset_list:
        for NUM_PARTS in num_parts_list:
            combine_yaml = combine_yaml_head(DATA_SET, NUM_PARTS)

            for PART_ALGO in [ 'random', 'metis', 'vcrandom', 'vcoblivious', 'vcdeg', 'vcns']:
                for NUM_HOP in [ halo_hop ]:
                    start = time.time()
                    first_n = NUM_PARTS if save_first_n < 0 else save_first_n
                    RUN_ID=f"partition_{DATA_SET}_{NUM_PARTS}_{first_n}_{PART_ALGO}_{NUM_HOP}"
                    CMD=f"python3 /workspace/dgl_dsg/experiments/tools/preprocessing.py  --dataset {DATA_SET} --part-algo {PART_ALGO} --num-hops {NUM_HOP} --n-parts {NUM_PARTS} --one_id {RUN_ID}  --data-root-path {DATA_ROOT} --save-first-n-parts {save_first_n}"
                    os.system(CMD)
                    mins = (time.time() - start) / 60
                    if PART_ALGO in USED_ALGO_IN_COMBINE:
                        combine_yaml += f"\t - {DATA_SET}_{PART_ALGO}: {DATA_ROOT}/ds_pre/{DATA_SET}/data_part_n{NUM_PARTS}_{PART_ALGO}_{NUM_HOP}_{first_n}/{DATA_SET}.json"
                        combine_yaml += "\n"

                    print(f"{RUN_ID} time(mins) : {mins}")

            if halo_hop == 0:
                combine_yaml_file = f"/data/ds_pre/{DATA_SET}/part_n{NUM_PARTS}_combine_0_{first_n}.yaml"
                with open(combine_yaml_file, "w") as tmp:
                    tmp.write(combine_yaml)

run(['ogbpr', 'ogbpa'], [1, 4, 16, 64], save_first_n=-1)
run(['ogbpr', 'ogbpa'], [256, 1024], save_first_n=64)

