#! /usr/bin/python3

import argparse
import os

parser = argparse.ArgumentParser(
                    prog='ds4gnn_launch.py',
                    description='Data serving for GNN traning: train launch scripts',
                    epilog='Text at the bottom of help')

parser.add_argument('--one_id', type=str, required=True, help='a unique id to identify experiments setup')

parser.add_argument('--dataset', type=str, required=True, choices=['ogbpr', 'ogbpa', 'cora'], help='')
parser.add_argument('--part_algo', type=str, required=True, choices=['random', 'metis', 'vcrandom', 'vcoblivious', 'vcbgl', 'vcns', 'vcns_coverage', 'vcns_proximity', 'vcst', 'vcdeg', 'vcrgrp'], help='')
parser.add_argument('--part_hop', type=int, required=True, choices=[0, 1], help='')
parser.add_argument('--n_parts', type=int, required=True, help='')
parser.add_argument('--use_first_n_parts', type=int, required=True, default=-1, help='')
parser.add_argument('--model', type=str, required=True, choices=['sage', 'gcn', 'gat'], help='')
parser.add_argument('--layers', type=int, required=True, choices=[5, 4, 3, 2], help='')
parser.add_argument('--batch_size', type=int, required=True, help='')
parser.add_argument('--sampling', type=str, required=True, choices=['dft', 'bdr'], help='')

parser.add_argument('--n_epoch', type=int, required=True, help='number of epochs for training')
parser.add_argument('--n_mach', type=int, required=True, help='number of machines')
parser.add_argument('--n_gpu_per_mach', type=int, required=True, help='number of gpus per machine')
parser.add_argument('--n_server_per_mach', type=int, required=True, help='number of servers per machine')
parser.add_argument('--n_trainer_per_mach', type=int, required=True, help='number of trainer per machine')
parser.add_argument('--n_sampler_per_trainer', type=int, required=True, help='number of samplers per TRAINER')

parser.add_argument('--disable_backup_server', type=str, required=True,  choices=['True', 'False'], help='disable backup server or not')

parser.add_argument('--verbose', type=bool, required=False, help='print cmd line')

parser.add_argument("--resume_path", type=str, default=None, help="resume from a path of a checkpoint")
parser.add_argument("--checkpoint_path", type=str, default=None, help="a path to store all checkpoints")
parser.add_argument("--checkpoint_every", type=int, default=-1, help="save a checkpoint erver N EPOCHS")

parser.add_argument("--log_path", type=str, required=True, help="log path used for this script")

parser.add_argument("--eval_every", type=int, default=50, help="do evaluation every N epochs")

parser.add_argument("--ip_config", type=str, required=True, help="choose which ip config to use, this is useful to run multile 1-partition experiments")
args = parser.parse_args()

DEFAULT_FANOUT_OF_LAYER = {
    2: "25,10",
    3: "15,10,5",
    4: "20,15,10,5",
    5: "25,20,15,10,5"
}

CLASS_NUM_OF = {
    'ogbpr' : 47,
    'ogbpa' : 172,
    'cora'  : 7
}

N_PARTS= args.n_parts
FIRST_N= f"_first{args.use_first_n_parts}" if args.use_first_n_parts > 0 else ''

IP_CONF="wtool/ipconfigs/{}.txt".format(args.ip_config)
PART_CONF="DATA/ds_pre/{DS_NAME}/data_part_n{N_PARTS}_{PART_ALGO}_{PART_HOP}{FIRST_N}/{DS_NAME}.json".format(
	DS_NAME=args.dataset,
    N_PARTS=N_PARTS,
    PART_ALGO=args.part_algo,
    PART_HOP=args.part_hop,
    FIRST_N=FIRST_N)

TRAIN_LAUNCHER="python3 /workspace/work/ds4gnn/compiling/dgl_dsg/tools/launch.py --workspace /workspace/work/ds4gnn"
PYTHON_CMD_PATH="python3"

SINGLE_JOB_CMD = "{PYTHON_CMD_PATH} dgl_exp_th/train_dist.py".format(PYTHON_CMD_PATH=PYTHON_CMD_PATH)
#SINGLE_JOB_CMD  = "gdb -x gdb.txt --batch --args {PYTHON_CMD_PATH} dgl_exp_th/train_dist.py".format(PYTHON_CMD_PATH=PYTHON_CMD_PATH)
SINGLE_JOB_CMD += " --model {MODEL} --num_layers {N_LAYER} --fan_out {FANOUT}".format(
                    MODEL=args.model, 
                    N_LAYER=args.layers,
                    FANOUT=DEFAULT_FANOUT_OF_LAYER[args.layers]
                )
SINGLE_JOB_CMD += " --n_classes {N_CLASS} --graph_name {DS_NAME}".format(
                    N_CLASS=CLASS_NUM_OF[args.dataset],
                    DS_NAME=args.dataset
                )

SINGLE_JOB_CMD += " --num_gpus {N_GPUS} --ip_config {IP_CONF} --part_config {PART_CONF}".format(
                    N_GPUS = args.n_gpu_per_mach,
                    IP_CONF=IP_CONF,
                    PART_CONF=PART_CONF
                )
SINGLE_JOB_CMD += " --num_epochs {N_EPOCH} --batch_size {BATCH_SIZE}".format(
                    N_EPOCH=args.n_epoch,
                    BATCH_SIZE=args.batch_size
                )
SINGLE_JOB_CMD += "{BORDER}{BACKUP_SERVER}".format(
                BORDER=" --stop_at_border" if args.sampling == "bdr" else "",
                BACKUP_SERVER=" --disable_backup_server" if args.disable_backup_server == 'True' else ""
                )

SINGLE_JOB_CMD += f" --eval_every {args.eval_every}"

if args.resume_path is not None:
    SINGLE_JOB_CMD += f" --resume_path {args.resume_path}"

if args.checkpoint_path is not None:
    SINGLE_JOB_CMD += f" --checkpoint_path {args.checkpoint_path}"

SINGLE_JOB_CMD += f" --checkpoint_every {args.checkpoint_every}"

LAUNCH_JOBS_CMD_PREFIX="{TRAIN_LAUNCHER} --num_servers {N_SERVER} --num_trainers {N_TRAINER} --num_samplers {N_SAMPLER}".format(
                        TRAIN_LAUNCHER=TRAIN_LAUNCHER,
                        N_SERVER=args.n_server_per_mach,
                        N_TRAINER=args.n_trainer_per_mach,
                        N_SAMPLER=args.n_sampler_per_trainer
                    )
LAUNCH_JOBS_CMD_PREFIX += " --part_config {PART_CONF} --ip_config {IP_CONF}".format(
                        PART_CONF=PART_CONF,
                        IP_CONF=IP_CONF
                    )
if args.verbose:
    print(LAUNCH_JOBS_CMD_PREFIX)
    print(SINGLE_JOB_CMD)

LAUNCH_JOBS_CMD="{} \"{}\"".format(LAUNCH_JOBS_CMD_PREFIX, SINGLE_JOB_CMD)

os.system( "{} 2>&1 | tee -a {}/{}.log".format(LAUNCH_JOBS_CMD, args.log_path, args.one_id) )
os.system( "echo {} >> {}/progress_{}.txt".format(args.one_id, args.log_path, args.ip_config if args.ip_config is not None else '') )
