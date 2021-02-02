import os

# AVAILABLE_GPUS = [0, 1, 3, 5, 6, 7]

# for i, dataset in enumerate(["cub", "awa1", "awa2", "apy", "sun"]):
#     for exp in ["zsl_deep_ns_an", "zsl_deep_ns_no_an", "zsl_deep_no_ns_no_an", "zsl_deep_no_ns_an"]:
#         command = f"CUDA_VISIBLE_DEVICES={AVAILABLE_GPUS[i]} python slurm/run_zsl_hpo.py -e {exp} -d {dataset} -n 5"
#         # os.system(command)
#         print(command)

for dataset in ['cub', 'sun']:
    for exp in ['czsl_ours', 'czsl_default']:
        command = f"python slurm/run_hpo.py -d {dataset} -e {exp} -n 5 --pos_account"
        os.system(command)
        # print(command)
