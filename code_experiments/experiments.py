import os

import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import matplotlib.ticker as mtick
from QUBO_to_QAOA import build_qubo_matrix_with_slack, qubo_to_pauli
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

            Q, total_vars = build_qubo_matrix_with_slack(G, lambda_penalty=l)
            cost_hamiltonian = qubo_to_pauli(Q, total_vars)

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

            Q, total_vars = build_qubo_matrix_with_slack(G, lambda_penalty=l)
            cost_hamiltonian = qubo_to_pauli(Q, total_vars)

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
    import os
    colors = {'optimal':'tab:green', 'feasible':'tab:blue', 'invalid':'tab:red'}
    N = len(G.nodes)
    os.makedirs(save_dir, exist_ok=True)

    start = time.time()
    for l in lambda_values:
        # memorizza tutte le run
        all_probs = {p: {cat: [] for cat in colors} for p in p_values}

        for p in p_values:
            print(f"Running experiment for lambda={l}, p={p}")

            Q, total_vars = build_qubo_matrix_with_slack(G, lambda_penalty=l)
            cost_hamiltonian = qubo_to_pauli(Q, total_vars)

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
def experiment_cx_scaling(G, p_values, lambda_val, save_dir="./"):

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

    for p in p_values:
        print(f"Processing p={p} ...")

        Q, total_vars = build_qubo_matrix_with_slack(G, lambda_penalty=lambda_val)
        cost_hamiltonian = qubo_to_pauli(Q, total_vars)

        qaoa_ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
        qaoa_ansatz.measure_all()

        cx_counts_before.append(
            qaoa_ansatz.decompose(reps=5).count_ops().get('cx', 0)
        )

        for name, backend in backends.items():
            pm = generate_preset_pass_manager(
                optimization_level=1,
                backend=backend,
                seed_transpiler=42
            )

            transpiled = pm.run(qaoa_ansatz)
            cx_count = transpiled.decompose(reps=3).count_ops().get('cx', 0)
            cx_counts_after[name].append(cx_count)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(10,6))
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
    plt.ylabel("Number of Two-Qubit Gates")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
def compare_optimizers_qar(
    G,
    backend_factory,
    cost_func_estimator,
    p_values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    lambda_values=[10,20,50,70],
    optimizers=['COBYLA', 'Nelder-Mead', 'COBYQA', 'GA', 'PSO', 'Powell'],
    shots=0,
    n_runs=5,
    noise_model=None,
    save_dir='./'
):
    os.makedirs(save_dir, exist_ok=True)
    N = len(G.nodes)

    # seed per evolutivi
    np.random.seed(42)   # opzionale, per riproducibilità
    seeds = np.random.randint(0, 10000, size=n_runs)

    # struttura: optimizer -> p -> lambda -> lista QAR
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for opt in optimizers:
        print(f"Running optimizer: {opt}")

        for l in lambda_values:
            for p in p_values:
                print(f"λ={l}, p={p}")

                # Build QUBO e Hamiltonian
                Q, total_vars = build_qubo_matrix_with_slack(G, lambda_penalty=l)

                # Convert dict Q in matrice numpy
                Q_matrix = np.zeros((total_vars, total_vars))
                for (i,j), val in Q.items():
                    if isinstance(i,int) and isinstance(j,int):
                        Q_matrix[i,j] = val
                        Q_matrix[j,i] = val

                cost_hamiltonian = qubo_to_pauli(Q, total_vars)

                for run in range(n_runs):

                    qaoa_ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
                    qaoa_ansatz.measure_all()

                    pm_opts = {'optimization_level':3}
                    if noise_model is not None:
                        pm_opts['basis_gates'] = noise_model.basis_gates

                    pm = generate_preset_pass_manager(**pm_opts, backend=backend_factory)
                    qaoa_ansatz = pm.run(qaoa_ansatz)

                    init_params = np.random.rand(qaoa_ansatz.num_parameters) * np.pi

                    # -------- SEED handling --------
                    if opt in ['GA','PSO']:
                        seed = seeds[run]  # un seed diverso per ogni run
                    else:
                        seed = None

                    # -------- OPTIMIZATION --------
                    if opt in ['COBYLA', 'Nelder-Mead', 'COBYQA', 'Powell']:
                        with Session(backend=backend_factory) as session:
                            estimator = Estimator(mode=session)
                            estimator.options.default_shots = shots

                            result = minimize(
                                cost_func_estimator,
                                init_params,
                                args=(qaoa_ansatz, estimator, cost_hamiltonian),
                                method=opt,
                                options={'maxiter':50, 'disp':False}
                            )
                        params_opt = result.x

                    elif opt == 'GA':
                        def cost_evovaq(params):
                            return cost_func_estimator(params, qaoa_ansatz, backend_factory, cost_hamiltonian)

                        problem = Problem(
                            qaoa_ansatz.num_parameters,
                            [(-np.pi, np.pi)]*qaoa_ansatz.num_parameters,
                            cost_evovaq
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

                        res = global_search.optimize(
                            problem,
                            10,
                            max_gen=10,
                            verbose=False,
                            seed=seed
                        )
                        params_opt = res.x

                    elif opt == 'PSO':
                        def cost_evovaq(params):
                            return cost_func_estimator(params, qaoa_ansatz, backend_factory, cost_hamiltonian)

                        problem = Problem(
                            qaoa_ansatz.num_parameters,
                            [(-np.pi, np.pi)]*qaoa_ansatz.num_parameters,
                            cost_evovaq
                        )

                        optimizer = PSO(
                            vmin=-1.0,
                            vmax=1.0
                        )

                        res = optimizer.optimize(
                            problem,
                            10,
                            max_gen=10,
                            verbose=False,
                            seed=seed
                        )
                        params_opt = res.x

                    # -------- SAMPLING --------
                    optimized_circuit = qaoa_ansatz.assign_parameters(params_opt)

                    with Session(backend=backend_factory) as session:
                        sampler = Sampler(mode=session)
                        sampler.options.default_shots = shots
                        job = sampler.run([optimized_circuit])
                        counts_bin = job.result()[0].data.meas.get_counts()

                    # -------- QAR dai counts --------
                    total_counts = sum(counts_bin.values())
                    expected_value = 0.0

                    for bs, count in counts_bin.items():
                        x = np.array([int(b) for b in bs[-total_vars:]])
                        expected_value += (x.T @ Q_matrix @ x) * (count / total_counts)

                    # minimo globale QUBO
                    if total_vars <= 16:
                        all_x = np.array(list(product([0,1], repeat=total_vars)))
                        Cmin = np.min(np.sum(all_x @ Q_matrix * all_x, axis=1))
                    else:
                        def qubo_cost(x):
                            x_bin = np.array(x).round()
                            return x_bin.T @ Q_matrix @ x_bin
                        res_min = minimize(qubo_cost, np.random.rand(total_vars), method='COBYLA')
                        Cmin = qubo_cost(res_min.x)

                    qar = Cmin / expected_value  # minimizzazione
                    results[opt][p][l].append(qar)

    # ===============================
    # BEST λ per p e media sulle run
    # ===============================
    final_data = {}
    for opt in optimizers:
        qar_per_p = []
        for p in p_values:
            lambda_means = [np.mean(results[opt][p][l]) for l in lambda_values]
            best_lambda_val = max(lambda_means)  # best λ per questo p
            qar_per_p.append(best_lambda_val)
        final_data[opt] = qar_per_p

    # ===============================
    # BOXPLOT
    # ===============================
    plt.figure(figsize=(10,6))
    data = [final_data[opt] for opt in optimizers]
    box = plt.boxplot(data, labels=optimizers)
    for median in box['medians']:
        median.set_linewidth(2.5)

    plt.ylabel('Quantum Approximation Ratio')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"optimizer_comparison_{shots}_nonoise.pdf"))
    plt.show()

    return final_data