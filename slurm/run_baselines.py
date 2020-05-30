import os

for dataset in ["cub", "awa", "sun"]:
    for method in ["basic", "joint", "agem", "ewc_online", "mas"]:
        command = f"python slurm/run_hpo.py -c {method} -d {dataset} -e baseline -n 5"
        os.system(command)
