# =========================================================
# IMPORTS
# =========================================================
from power_system_graphs import build_graph
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import os
from evovaq.problem import Problem
from evovaq.ParticleSwarmOptimization import PSO
import seaborn as sns

# =========================================================
# GRAPH
# =========================================================
G = build_graph("case24_ieee_rts")
nodes = list(G.nodes())
n = len(nodes)
node_to_idx = {node: i for i, node in enumerate(nodes)}


# =========================================================
# FIXED QUBO STRUCTURE (NO λ HERE)
# =========================================================
def build_terms(G):
    terms = []

    for i in G.nodes():
        ii = node_to_idx[i]
        terms.append((1.0, ii, ii))

    for i in G.nodes():
        S = [i] + list(G.neighbors(i))
        S_idx = [node_to_idx[j] for j in S]

        for a in S_idx:
            terms.append((-1.0, a, a))

        for a in S_idx:
            for b in S_idx:
                if a < b:
                    terms.append((2.0, a, b))

    return terms


terms = build_terms(G)


# =========================================================
# COST FUNCTION (λ IS HERE)
# =========================================================
def classical_cost(bitstring, lam):
    x = np.array(bitstring).astype(int)

    cost = np.sum(x)

    for i in G.nodes():
        S = [i] + list(G.neighbors(i))
        S_idx = [node_to_idx[j] for j in S]

        cost += lam * (1 - np.sum(x[S_idx])) ** 2

    return cost


def is_valid(bitstring):
    x = np.array(bitstring).astype(int)

    for i in G.nodes():
        S = [i] + list(G.neighbors(i))
        S_idx = [node_to_idx[j] for j in S]

        if np.sum(x[S_idx]) == 0:
            return False
    return True


def categorize(bitstring, min_pm):
    x = np.array(bitstring).astype(int)
    ones = np.sum(x)

    if not is_valid(x):
        return "invalid"
    elif ones == min_pm:
        return "optimal"
    else:
        return "feasible"


# =========================================================
# QAOA
# =========================================================
def cost_layer(gamma, terms, lam):
    # --- struttura (come prima) ---
    for c, i, j in terms:
        if i == j:
            qml.RZ(2 * gamma * c, wires=i)
        else:
            qml.CNOT([i, j])
            qml.RZ(2 * gamma * c, wires=j)
            qml.CNOT([i, j])

    # --- constraint term (λ entra qui) ---
    for i in G.nodes():
        S = [i] + list(G.neighbors(i))
        S_idx = [node_to_idx[j] for j in S]

        # linear terms
        for a in S_idx:
            qml.RZ(2 * gamma * lam * (-2), wires=a)

        # quadratic terms
        for a in S_idx:
            for b in S_idx:
                if a < b:
                    qml.CNOT([a, b])
                    qml.RZ(2 * gamma * lam * (2), wires=b)
                    qml.CNOT([a, b])



def mixer_layer(beta):
    # local mixing
    for i in range(n):
        qml.RX(2 * beta, wires=i)

    alpha = 0.08

    for i in G.nodes():
        neighbors = list(G.neighbors(i))
        deg = len(neighbors)

        for j in neighbors:
            if i < j:  # evita doppioni
                strength = alpha * beta / max(deg, 1)
                qml.IsingXX(2 * strength, wires=[node_to_idx[i], node_to_idx[j]])
                

def qaoa_layer(gamma, beta, terms, lam):
    cost_layer(gamma, terms, lam)
    mixer_layer(beta)


# =========================================================
# SETTINGS
# =========================================================
p_values = [1, 2, 3, 4, 5]
lambda_values = [1]
rng = np.random.default_rng(42)
seeds = rng.integers(0, 10**9, size=5)

results = {
    lam: {
        p: {"optimal": [], "feasible": [], "invalid": []}
        for p in p_values
    }
    for lam in lambda_values
}

save_dir = "./experiments/PUBO_scaling/24_graph_aware_mixer/"
os.makedirs(save_dir, exist_ok=True)

population_size = 20
max_gen = 70


# =========================================================
# MAIN LOOP (λ → p → seed)
# =========================================================
for lam in lambda_values:

    print(f"\n=== Lambda (TUNING) = {lam} ===")

    for p in p_values:

        seed_results = []

        for seed in seeds:

            dev = qml.device("lightning.gpu", wires=n, shots=2048)
            #dev = qml.device("default.qubit", wires=n, shots=2048)

            @qml.qnode(dev)
            def sampler(params):
                gammas = params[:p]
                betas = params[p:]

                for i in range(n):
                    qml.Hadamard(wires=i)

                for l in range(p):
                    qaoa_layer(gammas[l], betas[l], terms, lam)

                return qml.sample(wires=range(n))


            def fitness(params):
                samples = sampler(params)
                return -np.mean([classical_cost(s, lam) for s in samples])


            problem = Problem(
                2 * p,
                (-np.pi, np.pi),
                fitness
            )

            pso = PSO(vmin=-0.4, vmax=0.4)

            res = pso.optimize(
                problem,
                population_size,
                max_gen=max_gen,
                verbose=False,
                seed=int(seed)
            )

            best_params = res.x
            samples = sampler(best_params)

            counts = {"optimal": 0, "feasible": 0, "invalid": 0}

            for s in samples:
                cat = categorize(s, min_pm=8)
                counts[cat] += 1

            total = len(samples)

            seed_results.append({
                "optimal": counts["optimal"] / total,
                "feasible": counts["feasible"] / total,
                "invalid": counts["invalid"] / total
            })

        # =========================================================
        # MEDIA SU SEED
        # =========================================================
        for cat in ["optimal", "feasible", "invalid"]:
            results[lam][p][cat].append(
                np.mean([r[cat] for r in seed_results])
            )

    # =========================================================
    # PLOT  FOR ALL LAMBDA
    # =========================================================
    feasible, optimal, invalid = [], [], []

    for p in p_values:
        feasible.append(np.mean(results[lam][p]["feasible"]))
        optimal.append(np.mean(results[lam][p]["optimal"]))
        invalid.append(np.mean(results[lam][p]["invalid"]))

    x = np.arange(len(p_values))

    plt.figure()
    plt.plot(x, feasible, label="feasible")
    plt.plot(x, optimal, label="optimal")
    plt.plot(x, invalid, label="invalid")

    plt.xticks(x, p_values)
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    plt.title(f"lambda = {lam}")

    plt.savefig(os.path.join(save_dir, f"lambda_{lam}.pdf"))
    plt.close()


# =========================================================
# 🔥 HEATMAP GLOBALE (ALLA FINE)
# =========================================================
optimal_map = np.zeros((len(lambda_values), len(p_values)))
feasible_map = np.zeros((len(lambda_values), len(p_values)))
invalid_map  = np.zeros((len(lambda_values), len(p_values)))

for i, lam in enumerate(lambda_values):
    for j, p in enumerate(p_values):

        optimal_map[i, j] = np.mean(results[lam][p]["optimal"])
        feasible_map[i, j] = np.mean(results[lam][p]["feasible"])
        invalid_map[i, j]  = np.mean(results[lam][p]["invalid"])


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(optimal_map, xticklabels=p_values, yticklabels=lambda_values,
            cmap="viridis", ax=axes[0])
axes[0].set_title("Optimal")

sns.heatmap(feasible_map, xticklabels=p_values, yticklabels=lambda_values,
            cmap="viridis", ax=axes[1])
axes[1].set_title("Feasible")

sns.heatmap(invalid_map, xticklabels=p_values, yticklabels=lambda_values,
            cmap="viridis", ax=axes[2])
axes[2].set_title("Invalid")

plt.tight_layout()

plt.savefig(os.path.join(save_dir, "heatmap_global.pdf"))
plt.close()