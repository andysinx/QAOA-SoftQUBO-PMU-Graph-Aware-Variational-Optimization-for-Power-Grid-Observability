import pandapower.networks as pn
import networkx as nx

networks = {
    "case5": pn.case5,
    "case9": pn.case9,
    "case14": pn.case14,
    "case24_ieee_rts": pn.case24_ieee_rts,
    "case30": pn.case30,
    "case39": pn.case39,
    "case57": pn.case57,
    "case118": pn.case118,
}

def build_graph(case_name):
    net = networks[case_name]()
    G = nx.Graph()

    for bus in net.bus.index:
        G.add_node(bus)

    for _, line in net.line.iterrows():
        G.add_edge(line.from_bus, line.to_bus)

    for _, trafo in net.trafo.iterrows():
        G.add_edge(trafo.hv_bus, trafo.lv_bus)

    return G


def get_neighbors(G):
    return {n: sorted(G.neighbors(n)) for n in G.nodes}