import numpy as np
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
# ESPERIMENTO 1 (GENERALIZZATO)
# -----------------------------
def experiment_prob_vs_p(
    G,
    neighbors,
    is_valid,
    gpu_instance,
    cost_func_estimator,
    p_values,
    lambda_values,
    min_pm
):
    colors = {'optimal':'tab:green', 'feasible':'tab:blue', 'invalid':'tab:red'}
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
                backend=gpu_instance,
                seed_transpiler=42
            )
            qaoa_ansatz = pm.run(qaoa_ansatz)

            init_params = np.random.rand(qaoa_ansatz.num_parameters)*np.pi

            with Session(backend=gpu_instance) as session:
                estimator = Estimator(mode=session)
                estimator.options.default_shots = 0

                result = minimize(
                    cost_func_estimator,
                    init_params,
                    args=(qaoa_ansatz, estimator, cost_hamiltonian),
                    method='COBYLA',
                    options={'maxiter':50, 'disp':False}
                )

            optimized_circuit = qaoa_ansatz.assign_parameters(result.x)

            with Session(backend=gpu_instance) as session:
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
        plt.savefig(f"plot_{l}.pdf")


     


# -----------------------------
# ESPERIMENTO 2 (GENERALIZZATO)
# -----------------------------
def experiment_cx_scaling(G, p_values, lambda_val):

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