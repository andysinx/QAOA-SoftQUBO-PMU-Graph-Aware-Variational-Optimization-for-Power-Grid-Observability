import numpy as np
import math
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from evovaq.problem import Problem
from evovaq.GeneticAlgorithm import GA
from evovaq.HillClimbing import HC
from evovaq.MemeticAlgorithm import MA
import evovaq.tools.operators as op

# ------------------------
# PMU observation
# ------------------------
def pmu_observation(G, pmu_nodes):
    observed = set()
    for n in pmu_nodes:
        observed.add(n)
        observed.update(G.neighbors(n))
    return observed


# ------------------------
# QUBO
# ------------------------
def build_qubo_matrix_with_slack(G, lambda_penalty=50):
    Q = {}
    nodes = list(G.nodes)
    N = len(nodes)

    def add(i, j, value):
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0) + value

    slack_bits = {}
    current_idx = N

    for i in nodes:
        deg_i = len(list(G.neighbors(i))) + 1
        num_bits = math.ceil(math.log2(deg_i + 1))

        slack_bits[i] = []
        for _ in range(num_bits):
            slack_bits[i].append(current_idx)
            current_idx += 1

    total_vars = current_idx

    # Objective
    for i in nodes:
        add(i, i, 1)

    # Constraints
    for i in nodes:
        neighbors_i = list(G.neighbors(i))
        terms = []

        for a in [i] + neighbors_i:
            terms.append((a, 1))

        for k, var in enumerate(slack_bits[i]):
            terms.append((var, 2**k))

        N_i = len(neighbors_i) + 1

        add("const", "const", lambda_penalty * (N_i ** 2))

        for var, w in terms:
            add(var, var, lambda_penalty * (w**2 - 2 * N_i * w))

        for idx_a in range(len(terms)):
            var_a, w_a = terms[idx_a]
            for idx_b in range(idx_a + 1, len(terms)):
                var_b, w_b = terms[idx_b]
                add(var_a, var_b, 2 * lambda_penalty * w_a * w_b)

    return Q, total_vars


# ------------------------
# QUBO → Pauli
# ------------------------
def qubo_to_pauli(Q, N):
    pauli_dict = {}
    constant = 0

    def add_term(pauli_str, coeff):
        pauli_dict[pauli_str] = pauli_dict.get(pauli_str, 0) + coeff

    for (i, j), coeff in Q.items():
        if i == "const":
            constant += coeff
            continue

        if i == j:
            z = ['I'] * N
            z[i] = 'Z'
            add_term("".join(z), -coeff / 2)
            constant += coeff / 2

        else:
            z_i = ['I'] * N
            z_i[i] = 'Z'
            add_term("".join(z_i), -coeff / 4)

            z_j = ['I'] * N
            z_j[j] = 'Z'
            add_term("".join(z_j), -coeff / 4)

            z_ij = ['I'] * N
            z_ij[i] = 'Z'
            z_ij[j] = 'Z'
            add_term("".join(z_ij), coeff / 4)

            constant += coeff / 4

    add_term("I" * N, constant)
    return SparsePauliOp.from_list(list(pauli_dict.items()))

def define_backend(use_noise = False):
    if use_noise:
            backend_fake = FakeWashingtonV2()
            noise_model = NoiseModel.from_backend(backend_fake)
            gpu_instance = AerSimulator(noise_model=noise_model, method="statevector", device="GPU")
            gpu_instance.set_options(precision='single')
            return noise_model, gpu_instance
    else:
            gpu_instance = AerSimulator(method="statevector", device="GPU")
    return gpu_instance

def cost_func_estimator(params, circuit, estimator, observable):
    objective_func_vals = []
    pubs = [(circuit, observable, params)]
    result = estimator.run(pubs).result()
    cost_mean_val = result[0].data.evs 
    objective_func_vals.append(cost_mean_val)
    return cost_mean_val

def is_valid(bitstring, neighbors):
    """
    Verify if select configuration is ok for observer all networks.

    Parameters:
    - bitstring: string type '01011'
    - neighbors: dict {node: [neighbors]}

    Returns:
    - True / False
    """
    for i, val in enumerate(bitstring):
        if val == '0':
            # nodo i NON ha PMU → deve essere coperto da almeno un vicino
            if all(bitstring[j] == '0' for j in neighbors[i]):
                return False
    return True


'''# ------------------------
# Run QAOA
# ------------------------
def run_qaoa(G, backend, p=2, lambda_penalty=50):
    Q, total_vars = build_qubo_matrix_with_slack(G, lambda_penalty)
    cost_hamiltonian = qubo_to_pauli(Q, total_vars)

    qaoa_ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
    qaoa_ansatz.measure_all()

    np.random.seed(42)
    init_params = np.random.rand(qaoa_ansatz.num_parameters) * np.pi

    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = 0
        estimator.options.seed_estimator = 42

        result = minimize(
            cost_func_estimator,
            init_params,
            args=(qaoa_ansatz, estimator, cost_hamiltonian),  # <-- osservabily included
            method="COBYLA",
            options={"maxiter": 100, "disp": True}
        )

    optimized_circuit = qaoa_ansatz.assign_parameters(result.x)

    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)
        sampler.options.default_shots = 0 

        pub = (optimized_circuit,)
        job = sampler.run([pub], shots=int(1e4))
        counts_int = job.result()[0].data.meas.get_int_counts()
        counts_bin = job.result()[0].data.meas.get_counts()
        shots = sum(counts_int.values())
        final_distribution_int = {key: val / shots for key, val in counts_int.items()}
        final_distribution_bin = {key: val / shots for key, val in counts_bin.items()}
        print('final distribution: ', final_distribution_int)




def run_qaoa_evovaq(G, backend, p=2, lambda_penalty=50, max_gen=10):
    # 1️⃣ Costruisci QUBO e Hamiltonian
    Q, total_vars = build_qubo_matrix_with_slack(G, lambda_penalty)
    cost_hamiltonian = qubo_to_pauli(Q, total_vars)

    # 2️⃣ Ansatz QAOA
    qaoa_ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
    qaoa_ansatz.measure_all()

    # 3️⃣ Definisci cost function compatibile con evovaq
    def cost_function_evovaq(params):
        with Session(backend=backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = 0
            pubs = [(qaoa_ansatz, cost_hamiltonian, params)]
            result = estimator.run(pubs).result()
            return result[0].data.evs

    # 4️⃣ Definisci problema evovaq
    num_params = qaoa_ansatz.num_parameters
    param_bounds = [(-np.pi, np.pi)] * num_params
    evovaq_problem = Problem(num_params, param_bounds, cost_function_evovaq)

    # 5️⃣ Ottimizzatore Memetic Algorithm
    global_search = GA(selection=op.sel_tournament,
                       crossover=op.cx_uniform,
                       mutation=op.mut_gaussian,
                       sigma=0.2, mut_indpb=0.15, cxpb=0.9, tournsize=5)

    def get_neighbour(problem, current_solution):
        neighbour = current_solution.copy()
        index = np.random.randint(0, len(current_solution))
        _min, _max = problem.param_bounds[index]
        neighbour[index] = np.random.uniform(_min, _max)
        return neighbour

    #local_search = HC(generate_neighbour=get_neighbour)

    optimizer_evovaq = MA(global_search=global_search.evolve_population,
                          sel_for_refinement=op.sel_best,
                          #local_search=local_search.stochastic_var,
                          frequency=0.1,
                          intensity=10,
                          seed=42)

    # 6️⃣ Esegui ottimizzazione
    res_evovaq = global_search.optimize(evovaq_problem, 10, max_gen=max_gen, verbose=True, seed=42)

    # 7️⃣ Crea circuito ottimizzato
    optimized_circuit = qaoa_ansatz.assign_parameters(res_evovaq.x)

    return optimized_circuit, res_evovaq'''