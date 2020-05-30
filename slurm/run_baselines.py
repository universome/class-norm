import os

for dataset in ["cub", "awa", "sun"]:
    for method in ["basic", "joint", "agem", "ewc", "mas"]:
        command = "python slurm/run_hpo.py -c {method} -d {dataset} -e baseline -n 1"
        os.system(command)
        break
    break
