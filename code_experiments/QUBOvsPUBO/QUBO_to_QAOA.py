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

# SOFT CONSTRAINTS (PUBO) without slack variables
def build_qubo_matrix_penalty_no_slack(G, lambda_penalty=50):
    Q = {}
    nodes = list(G.nodes)

    def add(i, j, value):
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0) + value

    # --------------------
    # Objective: sum x_i
    # --------------------
    for i in nodes:
        add(i, i, 1)

    # --------------------
    # Constraints
    # --------------------
    for i in nodes:
        S_i = [i] + list(G.neighbors(i))

        # linear terms
        for j in S_i:
            add(j, j, -lambda_penalty)

        # quadratic terms
        for a_idx in range(len(S_i)):
            a = S_i[a_idx]
            for b_idx in range(a_idx + 1, len(S_i)):
                b = S_i[b_idx]
                add(a, b, 2 * lambda_penalty)

        # constant term (optional, ignored in optimization but kept for completeness)
        Q[("const", "const")] = Q.get(("const", "const"), 0) + lambda_penalty

    return Q, len(nodes)


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

def define_backend1(use_noise=False):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit.providers.fake_provider import FakeWashingtonV2

    if use_noise:
        backend_fake = FakeWashingtonV2()
        noise_model = NoiseModel.from_backend(backend_fake)
        backend = AerSimulator(noise_model=noise_model, method="statevector", device="GPU")
        backend.set_options(precision='single')
        return backend, noise_model
    else:
        backend = AerSimulator(method="statevector", device="GPU")
        return backend, None

def cost_func_estimator(params, circuit, estimator, observable):
    objective_func_vals = []
    pubs = [(circuit, observable, params)]
    result = estimator.run(pubs).result()
    cost_mean_val = result[0].data.evs 
    objective_func_vals.append(cost_mean_val)
    return cost_mean_val

def cost_func_estimator_1(params, circuit, estimator, observable):
    pubs = [(circuit, observable, params)]
    result = estimator.run(pubs).result()
    return result[0].data.evs

def cost_func_estimator_gen(params, circuit, backend_factory, observable, shots=0):

    with Session(backend=backend_factory) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = shots

        pubs = [(circuit, observable, params)]
        result = estimator.run(pubs).result()

    return result[0].data.evs

def is_valid(bitstring, neighbors):
    """
    Verify if selected configuration is valid for observing all networks.

    Parameters:
    - bitstring: string like '01011'
    - neighbors: dict {node: [neighbors]}

    Returns:
    - True / False
    """
    for i, val in enumerate(bitstring):
        if val == '0':
            # node i has NO PMU → it must be covered by at least one neighbor
            if all(bitstring[j] == '0' for j in neighbors[i]):
                return False
    return True



# ------------------------
# PUBO with selective slack
# ------------------------
def build_pubo_with_selective_slack(G, k_threshold=5, lambda_penalty=50):
    Q = {}
    nodes = list(G.nodes)
    N = len(nodes)

    def add(i, j, value):
        if i == "const" or j == "const":
            key = (i, j)
        else:
            if i > j:
                i, j = j, i
            key = (i, j)

        Q[key] = Q.get(key, 0) + value

    # Slack variables only for "critical" nodes
    slack_vars = {}
    current_idx = N

    for i in nodes:
        k = len(list(G.neighbors(i))) + 1  # degree + itself

        if k >= k_threshold:
            # introduce slack ONLY here
            num_bits = math.ceil(math.log2(k + 1))
            slack_vars[i] = []

            for _ in range(num_bits):
                slack_vars[i].append(current_idx)
                current_idx += 1

    total_vars = current_idx

    # ------------------------
    # Objective (same style as yours)
    # ------------------------
    for i in nodes:
        add(i, i, 1)

    # ------------------------
    # Constraints
    # ------------------------
    for i in nodes:
        neighbors_i = list(G.neighbors(i))
        k = len(neighbors_i) + 1

        # ---------
        # CASE 1: small → direct
        # ---------
        if k < k_threshold:
            terms = []

            for a in [i] + neighbors_i:
                terms.append((a, 1))

            N_i = k

            add("const", "const", lambda_penalty * (N_i ** 2))

            for var, w in terms:
                add(var, var, lambda_penalty * (w**2 - 2 * N_i * w))

            for idx_a in range(len(terms)):
                var_a, w_a = terms[idx_a]
                for idx_b in range(idx_a + 1, len(terms)):
                    var_b, w_b = terms[idx_b]
                    add(var_a, var_b, 2 * lambda_penalty * w_a * w_b)

        # ---------
        # CASE 2: large → slack
        # ---------
        else:
            slack_list = slack_vars[i]

            # build symbolic product → replaced by slack
            # (simplified: slack represents the whole term)

            y = slack_list[0]  # first slack variable as representative

            # constraint: y ≈ product
            terms = []

            for a in [i] + neighbors_i:
                terms.append((a, 1))

            # slack penalty
            N_i = k

            add("const", "const", lambda_penalty * (N_i ** 2))

            # penalty between slack and sum
            add(y, y, lambda_penalty * 1)
            add("const", y, -2 * lambda_penalty * N_i)

            for var, w in terms:
                add(var, var, lambda_penalty * (w**2))
                add(var, y, -2 * lambda_penalty * w)

            for idx_a in range(len(terms)):
                var_a, w_a = terms[idx_a]
                for idx_b in range(idx_a + 1, len(terms)):
                    var_b, w_b = terms[idx_b]
                    add(var_a, var_b, 2 * lambda_penalty * w_a * w_b)

    return Q, total_vars