from power_system_graphs import *
from experiments import *
from QUBO_to_QAOA import *

G = build_graph("case5")
neighbors = get_neighbors(G)

experiment_prob_vs_p(
    G=G,
    neighbors=neighbors,
    is_valid=is_valid,   # la tua funzione
    gpu_instance=define_backend(),
    cost_func_estimator=cost_func_estimator,
    p_values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    lambda_values=[2,5,10,20,50,70,100,125,150,200],
    min_pm=2
)

experiment_cx_scaling(
    G=G,
    p_values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    lambda_val=2
)