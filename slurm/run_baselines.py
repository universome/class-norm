import os

for dataset in ["cub", "awa", "sun"]:
    for method in ["basic", "agem", "ewc_online", "mas"]:
        command = f"python slurm/run_hpo.py -c {method} -d {dataset} -e baseline_new -n 5"
        os.system(command)
