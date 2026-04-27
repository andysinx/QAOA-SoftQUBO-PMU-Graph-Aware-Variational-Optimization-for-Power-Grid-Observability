import math
from itertools import combinations, product
from qiskit.quantum_info import SparsePauliOp
import networkx as nx



from itertools import combinations
import networkx as nx

# ------------------------
# Native PUBO (no slack, no threshold)
# ------------------------
def build_pubo_native(G, lambda_penalty=50):

    P = {}
    nodes = list(G.nodes)

    def add(term, value):
        term = tuple(sorted(term))
        P[term] = P.get(term, 0) + value

    # ------------------------
    # Objective (example: minimize number of PMUs)
    # ------------------------
    for i in nodes:
        add((i,), 1)

    # ------------------------
    # Constraints (full PUBO form)
    # ------------------------
    for i in nodes:
        neighbors = list(G.neighbors(i))
        S_i = [i] + neighbors

        k = len(S_i)

        # Full constraint:
        # (1 - sum x_j)^2 -> fully expanded as PUBO

        for r in range(0, k + 1):
            for subset in combinations(S_i, r):

                # combinatorial coefficient
                coeff = ((-1) ** r) * lambda_penalty

                if len(subset) == 0:
                    add(("const",), coeff)
                else:
                    add(subset, coeff)

    return P



# ------------------------
# Hybrid PUBO (products + slack only when needed)
# ------------------------
def build_pubo_hybrid(G, lambda_penalty=50, k_max=4):

    P = {}
    nodes = list(G.nodes)

    # slack variable counter
    current_idx = max(nodes) + 1
    slack_bits = {}

    def add(term, value):
        term = tuple(sorted(term))
        P[term] = P.get(term, 0) + value

    # ------------------------
    # Objective
    # ------------------------
    for i in nodes:
        add((i,), 1)

    # ------------------------
    # Constraints
    # ------------------------
    for i in nodes:
        neighbors = list(G.neighbors(i))
        S_i = [i] + neighbors

        # ------------------------
        # Case 1: PUBO product (ok)
        # ------------------------
        if len(S_i) <= k_max:

            k = len(S_i)

            for r in range(0, k + 1):
                for subset in combinations(S_i, r):
                    coeff = ((-1) ** r) * lambda_penalty

                    if len(subset) == 0:
                        add(("const",), coeff)
                    else:
                        add(subset, coeff)

        # ------------------------
        # Case 2: slack (problematic node)
        # ------------------------
        else:
            # assign a binary slack variable
            s_i = current_idx
            current_idx += 1
            slack_bits[i] = s_i

            # constraint: at least one node observed
            # quadratic form:
            # (1 - sum x_j - s_i)^2

            # constant term
            add(("const",), lambda_penalty)

            # linear terms
            for j in S_i:
                add((j,), -2 * lambda_penalty)

            add((s_i,), -2 * lambda_penalty)

            # quadratic terms
            for a in S_i:
                for b in S_i:
                    add((a, b), lambda_penalty)

            for j in S_i:
                add((j, s_i), 2 * lambda_penalty)

            # slack self-consistency
            add((s_i, s_i), lambda_penalty)

    return P, current_idx, slack_bits

# ------------------------
# PUBO → Pauli operator
# ------------------------
def pubo_to_pauli(P, N):

    pauli_dict = {}

    def add_term(pauli_str, coeff):
        pauli_dict[pauli_str] = pauli_dict.get(pauli_str, 0) + coeff

    for term, coeff in P.items():

        # ------------------------
        # constant term
        # ------------------------
        if term == ("const",):
            add_term("I" * N, coeff)
            continue

        k = len(term)

        # ------------------------
        # mapping:
        # x = (1 - Z)/2
        # ------------------------
        for pattern in product([0, 1], repeat=k):

            pauli = ['I'] * N
            factor = coeff / (2 ** k)

            for idx, bit in enumerate(pattern):
                if bit == 1:
                    pauli[term[idx]] = 'Z'
                    factor *= -1

            add_term("".join(pauli), factor)

    # add total constant
    add_term("I" * N, 0)

    return SparsePauliOp.from_list(list(pauli_dict.items()))