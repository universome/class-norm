from firelab.hpo import compute_hpo_vals_idx


def generate_experiments_from_hpo_grid(hpo_grid):
    experiments_vals_idx = compute_hpo_vals_idx(hpo_grid)
    experiments_vals = [{p: hpo_grid[p][i] for p, i in zip(hpo_grid.keys(), idx)} for idx in experiments_vals_idx]

    return experiments_vals
