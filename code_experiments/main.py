from power_system_graphs import *
from experiments import *
from QUBO_to_QAOA import *

G = build_graph("case5")
neighbors = get_neighbors(G)

experiment_prob_vs_p_seeds(
    G=G,
    neighbors=neighbors,
    is_valid=is_valid,   # la tua funzione
    backend_factory=define_backend(),
    cost_func_estimator=cost_func_estimator,
    p_values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    lambda_values=[2,5,10,20,50,70,100,125,150,200],
    min_pm=2,
    save_dir="./cobyla/shots_1024_nonoise/"
)

noise_model,backend_factory = define_backend(use_noise=True)
experiment_prob_vs_p_seeds_noise(
    G=G,
    neighbors=neighbors,
    is_valid=is_valid,   # la tua funzione
    backend_factory=backend_factory,
    noise_model=noise_model,
    cost_func_estimator=cost_func_estimator,
    p_values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    lambda_values=[5,10,30,50,70],
    min_pm=2,
    save_dir="./experiments/QUBO/case_5/cobyla/shots_1024_noise/"
)



'''experiment_cx_scaling(
    G=G,
    p_values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    lambda_val=2
)'''