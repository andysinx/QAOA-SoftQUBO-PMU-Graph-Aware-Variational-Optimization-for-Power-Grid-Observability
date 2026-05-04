import os
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import matplotlib.ticker as mtick
from HEPUBO_to_QAOA import pubo_to_pauli
from QUBO_to_QAOA import build_qubo_matrix_with_slack, qubo_to_pauli
from HEPUBO_to_QAOA import *
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator, SamplerV2 as Sampler
from scipy.optimize import minimize
from qiskit_ibm_runtime.fake_provider import (
    FakeAlmadenV2, FakeTorontoV2, FakeCairoV2,
    FakeBrooklynV2, FakeCambridgeV2, FakeSingaporeV2
)
from qiskit import transpile
from qiskit_aer.noise import NoiseModel
from evovaq.problem import Problem
from evovaq.GeneticAlgorithm import GA
from evovaq.HillClimbing import HC
from evovaq.MemeticAlgorithm import MA
from evovaq.ParticleSwarmOptimization import PSO
import evovaq.tools.operators as op
from itertools import product
import pulp

# -----------------------------
# Funzione di categorizzazione
# -----------------------------
def categorize_solution(bitstring, neighbors, min_pm, is_valid):
    if is_valid(bitstring, neighbors):
        if bitstring.count('1') == min_pm:
            return 'optimal'
        else:
            return 'feasible'
    else:
        return 'invalid'

# -----------------------------
# Experiment 1 - Cobyla
# -----------------------------
def experiment_prob_vs_p(
    G,
    neighbors,
    is_valid,
    backend_factory,
    cost_func_estimator,
    p_values,
    lambda_values,
    save_dir,
    min_pm
):
    import os
    colors = {'optimal':'tab:green', 'feasible':'tab:blue', 'invalid':'tab:red'}
    os.makedirs(save_dir, exist_ok=True)
    N = len(G.nodes)

    prob_vs_p = defaultdict(lambda: {'optimal':0, 'feasible':0, 'invalid':0})

    start = time.time()

    for l in lambda_values:
        for p in p_values:
            print(f"Running experiment for p={p}, lambda={l}")

            '''Q, total_vars = build_qubo_matrix_with_slack(G, lambda_penalty=l)
            cost_hamiltonian = qubo_to_pauli(Q, total_vars)'''

            P, N, _ = build_pubo_hybrid(G)
            cost_hamiltonian = pubo_to_pauli(P, N)

            qaoa_ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
            qaoa_ansatz.measure_all()

            pm = generate_preset_pass_manager(
                optimization_level=3,
                backend=backend_factory,
                seed_transpiler=42
            )
            qaoa_ansatz = pm.run(qaoa_ansatz)

            init_params = np.random.rand(qaoa_ansatz.num_parameters)*np.pi

            with Session(backend=backend_factory) as session:
                estimator = Estimator(mode=session)
                estimator.options.default_shots = 0

                result = minimize(
                    cost_func_estimator,
                    init_params,
                    args=(qaoa_ansatz, estimator, cost_hamiltonian),
                    method='COBYLA',
                    options={'maxiter': 50, 'disp': False}
                )

            optimized_circuit = qaoa_ansatz.assign_parameters(result.x)

            with Session(backend=backend_factory) as session:
                sampler = Sampler(mode=session)
                sampler.options.default_shots = 0
                job = sampler.run([optimized_circuit])
                counts_bin = job.result()[0].data.meas.get_counts()

            total_counts = sum(counts_bin.values())
            probs = {'optimal':0, 'feasible':0, 'invalid':0}

            for bs, count in counts_bin.items():
                cat = categorize_solution(bs[-N:], neighbors, min_pm, is_valid)
                probs[cat] += count / total_counts

            prob_vs_p[p] = probs

        end_total = time.time()
        print(f"\nTotal Time : {end_total-start:.2f} sec")

        # -----------------------------
        # Plot
        # -----------------------------
        plt.figure(figsize=(8,5))
        for cat in ['optimal','feasible','invalid']:
            plt.plot(
                p_values,
                [prob_vs_p[p][cat] for p in p_values],
                marker='o',
                linewidth=2.5,
                markersize=8,
                label=cat,
                color=colors[cat]
            )

        plt.xlabel('p')
        plt.ylabel('Probability to obtain a solution')
        plt.xticks(p_values)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"plot_lambda{l}.pdf"))

# -----------------------------
# Experiment 2 - Probabilities vs p with shots and multiple seeds without noise
# -----------------------------

def experiment_prob_vs_p_seeds(
    G,
    neighbors,
    is_valid,
    backend_factory,        # function that returns a simulated AER backend
    cost_func_estimator,
    p_values,
    lambda_values,
    min_pm,
    save_dir,
    shots=1024,
):
    import os
    colors = {'optimal':'tab:green', 'feasible':'tab:blue', 'invalid':'tab:red'}
    N = len(G.nodes)
    os.makedirs(save_dir, exist_ok=True)

    for l in lambda_values:
        # memorizza tutte le run
        all_probs = {p: {cat: [] for cat in colors} for p in p_values}

        for p in p_values:
            print(f"Running experiment for lambda={l}, p={p}")

            '''Q, total_vars = build_qubo_matrix_with_slack(G, lambda_penalty=l)
            cost_hamiltonian = qubo_to_pauli(Q, total_vars)'''
            P, N, _ = build_pubo_hybrid(G)
            cost_hamiltonian = pubo_to_pauli(P, N)

            qaoa_ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
            qaoa_ansatz.measure_all()

            pm = generate_preset_pass_manager(
                optimization_level=3,
                backend=backend_factory
            )
            qaoa_ansatz = pm.run(qaoa_ansatz)

            # init params casuali
            init_params = np.random.rand(qaoa_ansatz.num_parameters) * np.pi

            with Session(backend=backend_factory) as session:
                estimator = Estimator(mode=session)
                estimator.options.default_shots = shots  # solo shots, niente seed

                result = minimize(
                    cost_func_estimator,
                    init_params,
                    args=(qaoa_ansatz, estimator, cost_hamiltonian),
                    method='COBYLA',
                    options={'maxiter': 50, 'disp': False}
                )

            optimized_circuit = qaoa_ansatz.assign_parameters(result.x)

            with Session(backend=backend_factory) as session:
                sampler = Sampler(mode=session)
                sampler.options.default_shots = shots
                job = sampler.run([optimized_circuit])
                counts_bin = job.result()[0].data.meas.get_counts()

            total_counts = sum(counts_bin.values())
            probs = {'optimal':0, 'feasible':0, 'invalid':0}
            for bs, count in counts_bin.items():
                cat = categorize_solution(bs[-N:], neighbors, min_pm, is_valid)
                probs[cat] += count / total_counts

            # salva la probabilità
            for cat in colors:
                all_probs[p][cat].append(probs[cat])

        # -----------------------------
        # Plot con shading min-max
        # -----------------------------
        plt.figure(figsize=(8,5))
        for cat in colors:
            y_mean = [np.mean(all_probs[p][cat]) for p in p_values]
            y_min = [np.min(all_probs[p][cat]) for p in p_values]
            y_max = [np.max(all_probs[p][cat]) for p in p_values]

            plt.plot(p_values, y_mean, marker='o', linewidth=2.5, markersize=8, label=cat, color=colors[cat])
            plt.fill_between(p_values, y_min, y_max, color=colors[cat], alpha=0.2)

        plt.xlabel('p')
        plt.ylabel('Probability to obtain a solution')
        plt.xticks(p_values)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"plot_lambda{l}.pdf"))


# -----------------------------
# Experiment 3 - Probabilities vs p with shots and multiple seeds with noise
# -----------------------------

def experiment_prob_vs_p_seeds_noise(
    G,
    neighbors,
    is_valid,
    backend_factory,        # function that returns a simulated AER backend
    noise_model,
    cost_func_estimator,
    p_values,
    lambda_values,
    min_pm,
    save_dir,
    shots=1024,
):
    colors = {'optimal':'tab:green', 'feasible':'tab:blue', 'invalid':'tab:red'}
    N = len(G.nodes)
    os.makedirs(save_dir, exist_ok=True)

    start = time.time()
    for l in lambda_values:
        # memorizza tutte le run
        all_probs = {p: {cat: [] for cat in colors} for p in p_values}

        for p in p_values:
            print(f"Running experiment for lambda={l}, p={p}")

            '''Q, total_vars = build_qubo_matrix_with_slack(G, lambda_penalty=l)
            cost_hamiltonian = qubo_to_pauli(Q, total_vars)'''
            P, N, _ = build_pubo_hybrid(G)
            cost_hamiltonian = pubo_to_pauli(P, N)

            qaoa_ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
            qaoa_ansatz.measure_all()

            pm = generate_preset_pass_manager(
                optimization_level=3,
                basis_gates=noise_model.basis_gates,
                backend=backend_factory
            )
            qaoa_ansatz = pm.run(qaoa_ansatz)

            # init params casuali
            init_params = np.random.rand(qaoa_ansatz.num_parameters) * np.pi

            with Session(backend=backend_factory) as session:
                estimator = Estimator(mode=session)
                estimator.options.default_shots = shots  # solo shots, niente seed

                result = minimize(
                    cost_func_estimator,
                    init_params,
                    args=(qaoa_ansatz, estimator, cost_hamiltonian),
                    method='COBYLA',
                    options={'maxiter': 50, 'disp': False}
                )

            optimized_circuit = qaoa_ansatz.assign_parameters(result.x)

            with Session(backend=backend_factory) as session:
                sampler = Sampler(mode=session)
                sampler.options.default_shots = shots
                job = sampler.run([optimized_circuit])
                counts_bin = job.result()[0].data.meas.get_counts()

            total_counts = sum(counts_bin.values())
            probs = {'optimal':0, 'feasible':0, 'invalid':0}
            for bs, count in counts_bin.items():
                cat = categorize_solution(bs[-N:], neighbors, min_pm, is_valid)
                probs[cat] += count / total_counts

            # salva la probabilità
            for cat in colors:
                all_probs[p][cat].append(probs[cat])

        end_total = time.time()
        print(f"\nTotal Time : {end_total-start:.2f} sec")
        # -----------------------------
        # Plot con shading min-max
        # -----------------------------
        plt.figure(figsize=(8,5))
        for cat in colors:
            y_mean = [np.mean(all_probs[p][cat]) for p in p_values]
            y_min = [np.min(all_probs[p][cat]) for p in p_values]
            y_max = [np.max(all_probs[p][cat]) for p in p_values]

            plt.plot(p_values, y_mean, marker='o', linewidth=2.5, markersize=8, label=cat, color=colors[cat])
            plt.fill_between(p_values, y_min, y_max, color=colors[cat], alpha=0.2)

        plt.xlabel('p')
        plt.ylabel('Probability to obtain a solution')
        plt.xticks(p_values)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"plot_lambda{l}.pdf"))



# -----------------------------
# Experiment 5 - Count Two-Qbit Gate after and before transpilation
# -----------------------------
def experiment_cx_scaling(G, p_values, lambda_val=None, save_dir="./experiments/PUBO_efficently/"):

    os.makedirs(save_dir, exist_ok=True)

    backends = {
        'FakeMelbourneV2': FakeSingaporeV2(),
        'FakeCambridgeV2': FakeCambridgeV2(),
        'FakeBrooklynV2': FakeBrooklynV2(),
        'FakeAlmadenV2': FakeAlmadenV2(),
        'FakeTorontoV2': FakeTorontoV2(),
        'FakeCairoV2': FakeCairoV2()
    }

    cx_counts_before = []
    cx_counts_after = {name: [] for name in backends}

    for i, p in enumerate(p_values):
        print(f"Processing p={p} ...")

        # Costruzione circuito
        '''Q, total_vars = build_qubo_matrix_with_slack(G, lambda_penalty=lambda_val) 
        cost_hamiltonian = qubo_to_pauli(Q, total_vars)'''
        P = build_pubo_native(G)
        N = len(G.nodes)
        cost_hamiltonian = pubo_to_pauli(P, N)

        qaoa_ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
        qaoa_ansatz.measure_all()

        # Count BEFORE transpilation
        cx_counts_before.append(
            qaoa_ansatz.decompose(reps=5).count_ops().get('cx', 0)
        )

        # Salva circuito solo per il primo p
        if i == 0:
            qaoa_ansatz.decompose(reps=5).draw(output='mpl').savefig(
                os.path.join(save_dir, "after_trasp_circuit.pdf")
            )

        # Count AFTER per ogni backend
        for name, backend in backends.items():
            pm = generate_preset_pass_manager(
                optimization_level=1,
                backend=backend,
                seed_transpiler=42
            )

            transpiled = pm.run(qaoa_ansatz)

            cx_count = transpiled.decompose(reps=5).count_ops().get('cx', 0)
            cx_counts_after[name].append(cx_count)

            # Salva circuito transpiled solo per il primo p
            if i == 0 and name == list(backends.keys())[0]:
                transpiled.decompose(reps=5).draw(output='mpl').savefig(
                    os.path.join(save_dir, "before_trasp_circuit.pdf")
                )

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(
        p_values,
        cx_counts_before,
        marker='x',
        linestyle='-',
        linewidth=2.5,
        color='tab:gray',
        label='Before transpilation'
    )

    for name, counts in cx_counts_after.items():
        plt.plot(
            p_values,
            counts,
            marker='o',
            linestyle='-',
            linewidth=2.5,
            label=f'After transpilation ({name})'
        )

    plt.xticks(p_values)
    plt.xlabel("p")
    plt.ylabel("Number of Two-Qubit Gates (CX)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "count_twoqbit_gates.pdf"))
    plt.close()



def _default_dict_2():
    return defaultdict(list)

def _default_dict_3():
    return defaultdict(_default_dict_2)


def solve_qubo_milp(Q):
    n = Q.shape[0]

    # Problema di minimizzazione
    prob = pulp.LpProblem("QUBO_MILP", pulp.LpMinimize)

    # Variabili binarie x_i
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Variabili ausiliarie y_ij per linearizzare x_i x_j
    y = {}

    for i in range(n):
        for j in range(i + 1, n):
            y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", cat="Binary")

    # Obiettivo QUBO
    objective = 0

    for i in range(n):
        objective += Q[i, i] * x[i]

    for i in range(n):
        for j in range(i + 1, n):
            objective += Q[i, j] * y[(i, j)]

    prob += objective

    # Vincoli di linearizzazione
    for i in range(n):
        for j in range(i + 1, n):
            prob += y[(i, j)] <= x[i]
            prob += y[(i, j)] <= x[j]
            prob += y[(i, j)] >= x[i] + x[j] - 1

    # Risoluzione
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Estrai soluzione
    x_sol = np.array([pulp.value(var) for var in x])

    return x_sol, pulp.value(prob.objective)


# Per Cmax: basta massimizzare
def solve_qubo_milp_max(Q):
    n = Q.shape[0]

    prob = pulp.LpProblem("QUBO_MILP_MAX", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    y = {}

    for i in range(n):
        for j in range(i + 1, n):
            y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", cat="Binary")

    objective = 0

    for i in range(n):
        objective += Q[i, i] * x[i]

    for i in range(n):
        for j in range(i + 1, n):
            objective += Q[i, j] * y[(i, j)]

    prob += objective

    for i in range(n):
        for j in range(i + 1, n):
            prob += y[(i, j)] <= x[i]
            prob += y[(i, j)] <= x[j]
            prob += y[(i, j)] >= x[i] + x[j] - 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    return pulp.value(prob.objective)

def compare_optimizers_qar_qubo(
    G,
    backend_factory,
    cost_func_estimator,
    cost_func_estimator_gen,
    lambda_values,
    p_values,
    optimizers=['COBYLA', 'COBYQA', 'Nelder-Mead', 'GA', 'PSO'],
    shots=0,
    n_runs=20,
    noise_model=None,
    save_dir='./'
):
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from itertools import product
    from scipy.optimize import minimize

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"qar_results_{shots}_qubo_noise.pkl")

    results = defaultdict(_default_dict_3)

    np.random.seed(42)
    seeds = np.random.randint(0, 10000, size=n_runs)

    for opt in optimizers:
        print(f"Running optimizer: {opt}")

        for l in lambda_values:
            for p in p_values:
                print(f"λ={l}, p={p}")

                Q, total_vars = build_qubo_matrix_with_slack(G, lambda_penalty=l)

                Q_matrix = np.zeros((total_vars, total_vars))
                for (i, j), val in Q.items():
                    if isinstance(i, int) and isinstance(j, int):
                        Q_matrix[i, j] = val
                        Q_matrix[j, i] = val

                cost_hamiltonian = qubo_to_pauli(Q, total_vars)

                for run in range(n_runs):

                    qaoa_ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
                    qaoa_ansatz.measure_all()

                    pm_opts = {'optimization_level': 3}
                    if noise_model is not None:
                        pm_opts['basis_gates'] = noise_model.basis_gates

                    pm = generate_preset_pass_manager(**pm_opts, backend=backend_factory)
                    qaoa_ansatz = pm.run(qaoa_ansatz)

                    init_params = np.random.rand(qaoa_ansatz.num_parameters) * np.pi

                    if opt in ['COBYLA', 'Nelder-Mead', 'COBYQA', 'Powell']:
                        with Session(backend=backend_factory) as session:
                            estimator = Estimator(mode=session)
                            estimator.options.default_shots = shots

                            result = minimize(
                                cost_func_estimator,
                                init_params,
                                args=(qaoa_ansatz, estimator, cost_hamiltonian),
                                method=opt,
                                options={'maxiter': 50, 'disp': False}
                            )

                        params_opt = result.x

                    # =====================================================
                    # PSO (dal tuo codice QAOA)
                    # =====================================================
                    elif opt == 'PSO':

                        def fitness(params):
                            return -cost_func_estimator_gen(
                                params,
                                qaoa_ansatz,
                                backend_factory,
                                cost_hamiltonian,
                                shots
                            )

                        problem = Problem(
                            qaoa_ansatz.num_parameters,
                            (-np.pi, np.pi),
                            fitness
                        )

                        pso = PSO(vmin=-0.6, vmax=0.6)

                        res = pso.optimize(problem, 5, max_nfev=100, seed=seeds[run])

                        params_opt = res.x

                    # =====================================================
                    # GA (Memetic style come hai scritto tu)
                    # =====================================================
                    elif opt == 'GA':

                        def fitness(params):
                            return -cost_func_estimator_gen(
                                params,
                                qaoa_ansatz,
                                backend_factory,
                                cost_hamiltonian,
                                shots
                            )

                        problem = Problem(
                            qaoa_ansatz.num_parameters,
                            (-np.pi, np.pi),
                            fitness
                        )

                        global_search = GA(
                            selection=op.sel_tournament,
                            crossover=op.cx_uniform,
                            mutation=op.mut_gaussian,
                            sigma=0.2,
                            mut_indpb=0.15,
                            cxpb=0.9,
                            tournsize=5
                        )

                        def get_neighbour(problem, current_solution):
                            neighbour = current_solution.copy()
                            index = np.random.randint(0, len(current_solution))
                            _min, _max = problem.param_bounds[0]
                            neighbour[index] = np.random.uniform(_min, _max)
                            return neighbour

                        local_search = HC(generate_neighbour=get_neighbour)

                        optimizer = MA(
                            global_search=global_search.evolve_population,
                            sel_for_refinement=op.sel_best,
                            local_search=local_search.stochastic_var,
                            frequency=0.1,
                            intensity=10
                        )

                        res = optimizer.optimize(problem, 5, max_gen=10, verbose=False, seed=seeds[run])

                        params_opt = res.x

                    else:
                        
                        Exception(f"Optimizer {opt} not recognized. Skipping optimization.")

                    optimized_circuit = qaoa_ansatz.assign_parameters(params_opt)

                    with Session(backend=backend_factory) as session:
                        sampler = Sampler(mode=session)
                        sampler.options.default_shots = shots
                        job = sampler.run([optimized_circuit])
                        counts_bin = job.result()[0].data.meas.get_counts()

                    total_counts = sum(counts_bin.values())
                    expected_value = 0.0

                    for bs, count in counts_bin.items():
                        x = np.array([int(b) for b in bs[-total_vars:]])
                        expected_value += (x.T @ Q_matrix @ x) * (count / total_counts)

                    if total_vars <= 16:
                        all_x = np.array(list(product([0, 1], repeat=total_vars)))
                        costs = np.sum(all_x @ Q_matrix * all_x, axis=1)

                        Cmin = np.min(costs)
                        Cmax = np.max(costs)
                    else:
                        _, Cmin = solve_qubo_milp(Q_matrix)
                        Cmax = solve_qubo_milp_max(Q_matrix)

                    den = Cmax - Cmin

                    if abs(den) < 1e-12:
                        qar = 0.0
                    else:
                        qar = (expected_value - Cmin) / den

                    results[opt][p][l].append(qar)

    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved results to {save_path}")

    final_data = {}

    for opt in optimizers:
        qar_per_p = []
        for p in p_values:
            lambda_means = [np.mean(results[opt][p][l]) for l in lambda_values]
            best_lambda_val = max(lambda_means)
            qar_per_p.append(best_lambda_val)
        final_data[opt] = qar_per_p

    plt.figure(figsize=(10, 6))

    data = [final_data[opt] for opt in optimizers]

    box = plt.boxplot(
        data,
        labels=optimizers,
        showfliers=False,
        showmeans=True,
        meanline=True
    )

    for median in box['medians']:
        median.set_color('red')
        median.set_linewidth(2.5)

    for mean in box['means']:
        mean.set_color('blue')
        mean.set_linewidth(2.5)

    median_line = mlines.Line2D([], [], color='red', linewidth=2.5, label='Median')
    mean_line = mlines.Line2D([], [], color='blue', linewidth=2.5, label='Mean')

    plt.legend(handles=[median_line, mean_line])

    plt.ylabel('Quantum Approximation Ratio')
    plt.grid(True, axis='y')
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"optimizer_comparison_{shots}_qubo_noise.pdf"))
    plt.show()

    return final_data


def compare_optimizers_qar_pubo(
    G,
    backend_factory,
    cost_func_estimator,
    cost_func_estimator_gen,
    lambda_values,
    p_values,
    optimizers=['COBYLA', 'COBYQA', 'Nelder-Mead', 'GA', 'PSO'],
    shots=0,
    n_runs=20,
    noise_model=None,
    save_dir='./experiments/PUBO_efficiently/'
):

    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from itertools import product
    from scipy.optimize import minimize
    from collections import defaultdict

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"pubo_results_{shots}_noise.pkl")

    results = defaultdict(_default_dict_3)

    np.random.seed(42)
    seeds = np.random.randint(0, 10000, size=n_runs)

    # -----------------------------
    # PUBO cost (FIX CONST)
    # -----------------------------
    def evaluate_pubo(x, P):
        cost = 0.0
        for term, coeff in P.items():

            # termine costante
            if term == ("const",):
                cost += coeff
                continue

            if isinstance(term, int):
                cost += coeff * x[term]
            else:
                prod = 1
                for i in term:
                    prod *= x[i]
                cost += coeff * prod

        return cost

    for opt in optimizers:
        print(f"Running optimizer: {opt}")

        for l in lambda_values:
            for p in p_values:
                print(f"λ={l}, p={p}")

                # -----------------------------
                # PUBO BUILD (NATIVA)
                # -----------------------------
                P = build_pubo_native(G, lambda_penalty=l)
                total_vars = len(G.nodes)
                N = total_vars

                cost_hamiltonian = pubo_to_pauli(P, N)

                for run in range(n_runs):

                    qaoa_ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
                    qaoa_ansatz.measure_all()

                    pm_opts = {'optimization_level': 3}
                    if noise_model is not None:
                        pm_opts['basis_gates'] = noise_model.basis_gates

                    pm = generate_preset_pass_manager(**pm_opts, backend=backend_factory)
                    qaoa_ansatz = pm.run(qaoa_ansatz)

                    np.random.seed(seeds[run])
                    init_params = np.random.rand(qaoa_ansatz.num_parameters) * np.pi

                    # -----------------------------
                    # OPTIMIZATION
                    # -----------------------------
                    if opt in ['COBYLA', 'Nelder-Mead', 'COBYQA', 'Powell']:

                        with Session(backend=backend_factory) as session:
                            estimator = Estimator(mode=session)
                            estimator.options.default_shots = shots

                            result = minimize(
                                cost_func_estimator,
                                init_params,
                                args=(qaoa_ansatz, estimator, cost_hamiltonian),
                                method=opt,
                                options={'maxiter': 50, 'disp': False}
                            )

                        params_opt = result.x
                    
                    # =====================================================
                    # PSO (dal tuo codice QAOA)
                    # =====================================================
                    elif opt == 'PSO':

                        def fitness(params):
                            return -cost_func_estimator_gen(
                                params,
                                qaoa_ansatz,
                                backend_factory,
                                cost_hamiltonian,
                                shots
                            )

                        problem = Problem(
                            qaoa_ansatz.num_parameters,
                            (-np.pi, np.pi),
                            fitness
                        )

                        pso = PSO(vmin=-0.6, vmax=0.6)

                        res = pso.optimize(problem, 5, max_nfev=100, seed=seeds[run])

                        params_opt = res.x

                    # =====================================================
                    # GA (Memetic style come hai scritto tu)
                    # =====================================================
                    elif opt == 'GA':

                        def fitness(params):
                            return -cost_func_estimator_gen(
                                params,
                                qaoa_ansatz,
                                backend_factory,
                                cost_hamiltonian,
                                shots
                            )

                        problem = Problem(
                            qaoa_ansatz.num_parameters,
                            (-np.pi, np.pi),
                            fitness
                        )

                        global_search = GA(
                            selection=op.sel_tournament,
                            crossover=op.cx_uniform,
                            mutation=op.mut_gaussian,
                            sigma=0.2,
                            mut_indpb=0.15,
                            cxpb=0.9,
                            tournsize=5
                        )

                        def get_neighbour(problem, current_solution):
                            neighbour = current_solution.copy()
                            index = np.random.randint(0, len(current_solution))
                            _min, _max = problem.param_bounds[0]
                            neighbour[index] = np.random.uniform(_min, _max)
                            return neighbour

                        local_search = HC(generate_neighbour=get_neighbour)

                        optimizer = MA(
                            global_search=global_search.evolve_population,
                            sel_for_refinement=op.sel_best,
                            local_search=local_search.stochastic_var,
                            frequency=0.1,
                            intensity=10
                        )

                        res = optimizer.optimize(problem, 5, max_gen=10, verbose=False, seed=seeds[run])

                        params_opt = res.x



                    else:
                        
                        Exception(f"Optimizer {opt} not recognized. Skipping optimization.")

                    # -----------------------------
                    # SAMPLING (FIX COUNTS)
                    # -----------------------------
                    optimized_circuit = qaoa_ansatz.assign_parameters(params_opt)

                    with Session(backend=backend_factory) as session:
                        sampler = Sampler(mode=session)
                        sampler.options.default_shots = shots
                        job = sampler.run([optimized_circuit])

                        raw_counts = job.result()[0].data.meas.get_counts()

                        # FIX robusto
                        if isinstance(raw_counts, dict) and all(isinstance(v, dict) for v in raw_counts.values()):
                            counts_bin = list(raw_counts.values())[0]
                        else:
                            counts_bin = raw_counts

                    total_counts = sum(counts_bin.values())
                    expected_value = 0.0

                    for bs, count in counts_bin.items():
                        x = np.array([int(b) for b in bs[-total_vars:]])
                        expected_value += evaluate_pubo(x, P) * (count / total_counts)

                    # -----------------------------
                    # Cmin / Cmax
                    # -----------------------------
                    if total_vars <= 16:
                        all_x = np.array(list(product([0, 1], repeat=total_vars)))
                        costs = np.array([evaluate_pubo(x, P) for x in all_x])

                        Cmin = np.min(costs)
                        Cmax = np.max(costs)
                    else:
                        # 🔥 Cmin: local search discreta
                        def local_search(x0, P, n_iter=500, patience=50):
                            x = x0.copy()
                            best = evaluate_pubo(x, P)

                            no_improve = 0

                            for _ in range(n_iter):
                                i = np.random.randint(len(x))
                                x_new = x.copy()
                                x_new[i] = 1 - x_new[i]

                                val = evaluate_pubo(x_new, P)

                                if val < best:
                                    x = x_new
                                    best = val
                                    no_improve = 0
                                else:
                                    no_improve += 1

                                if no_improve >= patience:
                                    break

                            return best

                        x0 = (np.random.rand(total_vars) > 0.5).astype(int)
                        Cmin = local_search(x0, P)

                        # 🔥 Cmax: bound più corretto del tuo
                        Cmax = 0.0
                        for term, coeff in P.items():
                            if term == ("const",):
                                Cmax += abs(coeff)
                            else:
                                Cmax += abs(coeff)

                    den = Cmax - Cmin

                    if abs(den) < 1e-12:
                        qar = 0.0
                    else:
                        qar = (expected_value - Cmin) / den

                    results[opt][p][l].append(qar)

    # -----------------------------
    # SAVE
    # -----------------------------
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved results to {save_path}")

    # -----------------------------
    # BEST λ per p
    # -----------------------------
    final_data = {}

    for opt in optimizers:
        qar_per_p = []
        for p in p_values:
            lambda_means = [np.mean(results[opt][p][l]) for l in lambda_values]
            best_lambda_val = max(lambda_means)
            qar_per_p.append(best_lambda_val)
        final_data[opt] = qar_per_p

    # -----------------------------
    # PLOT (IDENTICO AL QUBO)
    # -----------------------------
    plt.figure(figsize=(10, 6))

    data = [final_data[opt] for opt in optimizers]

    box = plt.boxplot(
        data,
        labels=optimizers,
        showfliers=False,
        showmeans=True,
        meanline=True
    )

    for median in box['medians']:
        median.set_color('red')
        median.set_linewidth(2.5)

    for mean in box['means']:
        mean.set_color('blue')
        mean.set_linewidth(2.5)

    median_line = mlines.Line2D([], [], color='red', linewidth=2.5, label='Median')
    mean_line = mlines.Line2D([], [], color='blue', linewidth=2.5, label='Mean')

    plt.legend(handles=[median_line, mean_line])

    plt.ylabel('Quantum Approximation Ratio')
    plt.grid(True, axis='y')
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"optimizer_comparison_{shots}_pubo_noise.pdf"))
    plt.show()

    return final_data




def experiment_prob_vs_p_pubo(
    G,
    neighbors,
    is_valid,
    backend_factory,
    cost_func_estimator,
    p_values,
    lambda_values,
    min_pm,
    save_dir="./experiments/PUBO_efficently/cobyla/",
    shots=0,
    noise_model=None,
    n_runs=1,
    use_seeds=True
):
    import os
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from collections import defaultdict
    from scipy.optimize import minimize

    os.makedirs(save_dir, exist_ok=True)

    colors = {'optimal': 'tab:green', 'feasible': 'tab:blue', 'invalid': 'tab:red'}
    N = len(G.nodes)

    # seeds globali
    np.random.seed(42)
    seeds = np.random.randint(0, 10000, size=n_runs)

    start = time.time()

    for l in lambda_values:

        # salva tutte le run
        all_probs = {p: {cat: [] for cat in colors} for p in p_values}

        for p in p_values:
            print(f"Running experiment λ={l}, p={p}")

            # -----------------------------
            # PUBO NATIVE
            # -----------------------------
            P = build_pubo_native(G, lambda_penalty=l)
            cost_hamiltonian = pubo_to_pauli(P, N)

            # -----------------------------
            # CIRCUIT (costruito UNA VOLTA)
            # -----------------------------
            qaoa_ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
            qaoa_ansatz.measure_all()

            pm_opts = {
                'optimization_level': 3,
                'backend': backend_factory
            }

            if noise_model is not None:
                pm_opts['basis_gates'] = noise_model.basis_gates

            pm = generate_preset_pass_manager(**pm_opts)
            qaoa_ansatz = pm.run(qaoa_ansatz)

            # -----------------------------
            # RUN MULTIPLE
            # -----------------------------
            for run in range(n_runs):

                if use_seeds:
                    np.random.seed(seeds[run])

                init_params = np.random.rand(qaoa_ansatz.num_parameters) * np.pi

                # -----------------------------
                # OPTIMIZATION
                # -----------------------------
                with Session(backend=backend_factory) as session:
                    estimator = Estimator(mode=session)
                    estimator.options.default_shots = shots

                    result = minimize(
                        cost_func_estimator,
                        init_params,
                        args=(qaoa_ansatz, estimator, cost_hamiltonian),
                        method='COBYLA',
                        options={'maxiter': 50, 'disp': False}
                    )

                optimized_circuit = qaoa_ansatz.assign_parameters(result.x)

                # -----------------------------
                # SAMPLING
                # -----------------------------
                with Session(backend=backend_factory) as session:
                    sampler = Sampler(mode=session)
                    sampler.options.default_shots = shots
                    job = sampler.run([optimized_circuit])

                    raw_counts = job.result()[0].data.meas.get_counts()

                    # FIX robusto counts
                    if isinstance(raw_counts, dict) and all(isinstance(v, dict) for v in raw_counts.values()):
                        counts_bin = list(raw_counts.values())[0]
                    else:
                        counts_bin = raw_counts

                total_counts = sum(counts_bin.values())
                probs = {'optimal': 0, 'feasible': 0, 'invalid': 0}

                for bs, count in counts_bin.items():
                    bitstring = bs[-N:]
                    cat = categorize_solution(bitstring, neighbors, min_pm, is_valid)
                    probs[cat] += count / total_counts

                for cat in colors:
                    all_probs[p][cat].append(probs[cat])

        # -----------------------------
        # TIME
        # -----------------------------
        end_total = time.time()
        print(f"\nTotal Time (λ={l}): {end_total - start:.2f} sec")

        # -----------------------------
        # PLOT
        # -----------------------------
        plt.figure(figsize=(8, 5))

        for cat in colors:

            y_mean = [np.mean(all_probs[p][cat]) for p in p_values]

            if n_runs > 1:
                y_min = [np.min(all_probs[p][cat]) for p in p_values]
                y_max = [np.max(all_probs[p][cat]) for p in p_values]

                plt.fill_between(
                    p_values,
                    y_min,
                    y_max,
                    color=colors[cat],
                    alpha=0.2
                )

            plt.plot(
                p_values,
                y_mean,
                marker='o',
                linewidth=2.5,
                markersize=8,
                label=cat,
                color=colors[cat]
            )

        plt.xlabel('p')
        plt.ylabel('Probability to obtain a solution')
        plt.xticks(p_values)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        suffix = f"shots{shots}"
        if noise_model is not None:
            suffix += "_noise"
        if n_runs > 1:
            suffix += "_multirun"

        plt.savefig(os.path.join(save_dir, f"plot_lambda{l}_{suffix}.pdf"))
        plt.show()




def experiment_prob_vs_p_pubo_gen(
    G,
    neighbors,
    is_valid,
    backend_factory,
    cost_func_estimator,
    p_values,
    lambda_values,
    min_pm,
    save_dir="./experiments/PUBO_scaling/",
    shots=0,
    noise_model=None,
    n_runs=5,
    use_seeds=True
):
    import os
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    os.makedirs(save_dir, exist_ok=True)

    colors = {'optimal': 'tab:green', 'feasible': 'tab:blue', 'invalid': 'tab:red'}
    N = len(G.nodes)

    # seeds globali
    np.random.seed(42)
    seeds = np.random.randint(0, 10000, size=n_runs)

    start = time.time()

    for l in lambda_values:

        all_probs = {p: {cat: [] for cat in colors} for p in p_values}

        for p in p_values:
            print(f"Running experiment λ={l}, p={p}")

            # -----------------------------
            # PUBO NATIVE
            # -----------------------------
            P = build_pubo_native(G, lambda_penalty=l)
            cost_hamiltonian = pubo_to_pauli(P, N)

            # -----------------------------
            # CIRCUIT
            # -----------------------------
            qaoa_ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
            qaoa_ansatz.measure_all()

            pm_opts = {
                'optimization_level': 3,
                'backend': backend_factory
            }

            if noise_model is not None:
                pm_opts['basis_gates'] = noise_model.basis_gates

            pm = generate_preset_pass_manager(**pm_opts)
            qaoa_ansatz = pm.run(qaoa_ansatz)

            # -----------------------------
            # RUN MULTIPLE
            # -----------------------------
            for run in range(n_runs):

                if use_seeds:
                    np.random.seed(seeds[run])

                # -----------------------------
                # OPTIMIZATION (GA)
                # -----------------------------
                with Session(backend=backend_factory) as session:

                    estimator = Estimator(mode=session)
                    estimator.options.default_shots = shots

                    def fitness(params):
                        return -cost_func_estimator(
                            params,
                            qaoa_ansatz,
                            estimator,
                            cost_hamiltonian
                        )

                    problem = Problem(
                        qaoa_ansatz.num_parameters,
                        (-np.pi, np.pi),
                        fitness
                    )

                    ga_optimizer = GA(
                        selection=op.sel_tournament,
                        crossover=op.cx_uniform,
                        mutation=op.mut_gaussian,
                        sigma=0.2,
                        mut_indpb=0.15,
                        cxpb=0.9,
                        tournsize=5
                    )

                    res = ga_optimizer.optimize(
                        problem,
                        20,          
                        max_gen=50,
                        verbose=True,
                        seed=seeds[run] if use_seeds else 42
                    )

                optimized_circuit = qaoa_ansatz.assign_parameters(res.x)

                # -----------------------------
                # SAMPLING
                # -----------------------------
                with Session(backend=backend_factory) as session:
                    sampler = Sampler(mode=session)
                    sampler.options.default_shots = shots

                    job = sampler.run([optimized_circuit])
                    raw_counts = job.result()[0].data.meas.get_counts()

                    if isinstance(raw_counts, dict) and all(isinstance(v, dict) for v in raw_counts.values()):
                        counts_bin = list(raw_counts.values())[0]
                    else:
                        counts_bin = raw_counts

                total_counts = sum(counts_bin.values())
                probs = {'optimal': 0, 'feasible': 0, 'invalid': 0}

                for bs, count in counts_bin.items():
                    bitstring = bs[-N:]
                    cat = categorize_solution(bitstring, neighbors, min_pm, is_valid)
                    probs[cat] += count / total_counts

                for cat in colors:
                    all_probs[p][cat].append(probs[cat])

        # -----------------------------
        # TIME
        # -----------------------------
        end_total = time.time()
        print(f"\nTotal Time (λ={l}): {end_total - start:.2f} sec")

        # -----------------------------
        # PLOT
        # -----------------------------
        plt.figure(figsize=(8, 5))

        for cat in colors:

            y_mean = [np.mean(all_probs[p][cat]) for p in p_values]

            if n_runs > 1:
                y_min = [np.min(all_probs[p][cat]) for p in p_values]
                y_max = [np.max(all_probs[p][cat]) for p in p_values]

                plt.fill_between(
                    p_values,
                    y_min,
                    y_max,
                    color=colors[cat],
                    alpha=0.2
                )

            plt.plot(
                p_values,
                y_mean,
                marker='o',
                linewidth=2.5,
                markersize=8,
                label=cat,
                color=colors[cat]
            )

        plt.xlabel('p')
        plt.ylabel('Probability to obtain a solution')
        plt.xticks(p_values)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        suffix = f"shots{shots}"
        if noise_model is not None:
            suffix += "_noise"
        if n_runs > 1:
            suffix += "_multirun"

        plt.savefig(os.path.join(save_dir, f"plot_lambda{l}_p{p}_{suffix}.pdf"))
        plt.show()





