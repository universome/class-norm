import os

for i, dataset in enumerate(["cub", "awa1", "awa2", "apy" "sun"]):
    for exp in ["zsl_linear_ns_no_std", "zsl_linear_ns_std", "zsl_linear_no_ns_std", "zsl_linear_no_ns_no_std"]:
        command = f"CUDA_VISIBLE_DEVICES={i} python slurm/run_zsl_hpo.py -e {exp} -d {dataset} -n 5"
        os.system(command)
