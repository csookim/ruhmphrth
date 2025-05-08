import networkx as nx
import random
from qiskit.transpiler import Layout
import matplotlib.pyplot as plt
from qbraid import load_program
from qiskit import QuantumCircuit

def preprocessing(qc, only_cx=False):
    prg = load_program(qc)
    prg.remove_idle_qubits()
    qc = prg.program
    new_circuit = QuantumCircuit(qc.num_qubits, qc.num_qubits)
    qubit_map = {}
    for i, q in enumerate(qc.qubits):
        qubit_map[q] = new_circuit.qubits[i]

    for gate in qc.data:
        if gate.operation.name == 'measure':
            continue
        if gate.operation.name == 'barrier':
            continue
        if len(gate.qubits) == 2 and gate.operation.name != 'cx':
            print("error")
        if only_cx and gate.operation.name != 'cx':
            continue
        new_qubits = []
        for q in gate.qubits:
            new_qubits.append(qubit_map[q])
        new_circuit.append(gate.operation, new_qubits)
    new_circuit.measure(list(range(new_circuit.num_qubits)), list(range(new_circuit.num_qubits)))
    return new_circuit

def sub_iso_mapping(circ, backend):
    int_graph = qubit_interaction_graph(circ)
    logical_graph = nx.Graph()
    for (q1, q2), weight in int_graph.items():
        logical_graph.add_edge(q1, q2)

    physical_graph = nx.Graph()
    physical_graph.add_edges_from(backend.coupling_map)

    GM = nx.algorithms.isomorphism.GraphMatcher(physical_graph, logical_graph)
    matched = list(GM.subgraph_monomorphisms_iter())
    if len(matched) == 0:
        return None
    
    mapping = {}
    qubits = circ.qubits
    if len(matched) != 0:  
        for i, (ph, log) in enumerate(matched[0].items()):
            mapping[qubits[log]] = ph
            if qubits[log]._index != log:
                print("error")
    return mapping

def count_cx(circ):
    cx = 0
    for g in circ:
        if g.operation.name == 'cx':
            cx += 1
        if g.operation.name == 'swap':
            cx += 3
    return cx

def return_hw_map(circ):
    vqubits = set()
    for g in circ:
        for q in g.qubits:
            vqubits.add(q._index)
    
    return list(vqubits)

def qubit_interaction_graph(circuit, count=1e9):
    graph = {}
    cnt = 0
    for gate in circuit.data:
        if len(gate.qubits) == 2:
            args = tuple(sorted((gate.qubits[0]._index, gate.qubits[1]._index)))
            if cnt < count:
                graph[args] = graph.get(args, 0) + 1
            else:
                graph[args] = graph.get(args, 0)
            cnt += 1
    return graph

def print_map(layout, ancilla=False):
    new_layout = {}
    for k, v in layout.items():
        if k._register.name == "q":
            new_layout[k._index] = v
    new_new = sorted(new_layout.items(), key=lambda x: x[0])
    print(new_new)
    if ancilla:
        new_layout = {}
        for k, v in layout.items():
            if k._register.name == "ancilla":
                new_layout[k._index] = v
        new_new = sorted(new_layout.items(), key=lambda x: x[0])
        print(new_new)

def edge_density(graph_dict):
    nodes = set()
    for u, v in graph_dict.keys():
        nodes.add(u)
        nodes.add(v)
    
    n = len(nodes)
    num_edges = len(graph_dict)
    max_edges = n * (n - 1) / 2
    density = num_edges / max_edges
    return density

def draw_graph(graph_dict):
    g = nx.Graph()
    for k, v in graph_dict.items():
        g.add_edge(k[0], k[1], weight=v)

    pos = nx.spring_layout(g)
    nx.draw(g, pos, with_labels=True, node_color='lightblue', node_size=100, font_weight='bold')
    nx.draw_networkx_edges(g, pos)

    edge_labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)

    plt.title("Weighted Graph")
    plt.show()

def calculate_node_score(G, node, max_depth):
    score = G.degree(node)
    visited = set([node])
    current_depth_nodes = {node}
    
    for depth in range(max_depth + 1):
        next_depth_nodes = set()
        for current_node in current_depth_nodes:
            neighbors = set(G.neighbors(current_node))
            unvisited_neighbors = neighbors - visited
            next_depth_nodes.update(unvisited_neighbors)
            visited.update(unvisited_neighbors)
        
        sum = 0
        for next_node in next_depth_nodes:
            score += G.degree(next_node)
            sum += G.degree(next_node)
        # print("depth ", depth + 1, "sum: ", sum)
        # print("depth nodes: ", next_depth_nodes)
        current_depth_nodes = next_depth_nodes
        
        if not current_depth_nodes:
            break
    
    return score

def find_start_point(cmap):
    G = nx.Graph()
    G.add_edges_from(cmap)
    node_degrees = dict(G.degree())
    max_degree = max(node_degrees.values())
    max_degree_nodes = [node for node, degree in node_degrees.items() if degree == max_degree]
    random.shuffle(max_degree_nodes)
    # print("max degree nodes: ", max_degree_nodes)

    high_score = -1
    best_node = None
    for node in max_degree_nodes:
        hscore = calculate_node_score(G, node, 3)
        if hscore > high_score:
            high_score = hscore
            best_node = node
        # print("best node: ", best_node, " with score: ", high_score)
    return best_node

def generate_ordered_list_logical(logical_graph):
    node_weights = {}
    for node in logical_graph.nodes():
        weight_sum = sum(logical_graph[node][neighbor]['weight'] 
                         for neighbor in logical_graph.neighbors(node))
        node_weights[node] = weight_sum
    
    starting_node = max(node_weights, key=node_weights.get)
    ordered_list = [starting_node]
    visited = {starting_node}
    
    queue = [starting_node]
    while queue and len(ordered_list) < len(logical_graph):
        current = queue.pop(0)
        neighbors = [(n, node_weights[n]) for n in logical_graph.neighbors(current) 
                    if n not in visited]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        for neighbor, _ in neighbors:
            if neighbor not in visited:
                ordered_list.append(neighbor)
                visited.add(neighbor)
                queue.append(neighbor)
    
    return ordered_list

def generate_ordered_list_physical(physical_graph, starting_node):
    node_degrees = dict(physical_graph.degree())
    
    ordered_list = [starting_node]
    visited = {starting_node}
    
    queue = [starting_node]
    while queue and len(ordered_list) < len(physical_graph):
        current = queue.pop(0)
        neighbors = [(n, node_degrees[n]) for n in physical_graph.neighbors(current) 
                    if n not in visited]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        for neighbor, _ in neighbors:
            if neighbor not in visited:
                ordered_list.append(neighbor)
                visited.add(neighbor)
                queue.append(neighbor)
    
    return ordered_list

def update_layout(layout, mapping):
    vbits = set(layout.keys())
    pbits = set(layout.get_virtual_bits().values())

    mapping_vbits = set(mapping.keys())
    mapping_pbits = set(mapping.values())

    layout_map = {}
    layout_map.update(mapping)
    new_vbits = list(vbits - mapping_vbits)
    new_pbits = list(pbits - mapping_pbits)
    for i in range(len(new_vbits)):
        layout_map[new_vbits[i]] = new_pbits[i]
    # print(layout_map)

    new_layout = Layout()
    new_layout.from_dict(layout_map)
    return new_layout

def add_ancila(layout_anc, layout):
    vbits = set(layout_anc.keys())
    pbits = set(layout_anc.values())

    layout_vbits = set(layout.keys())
    layout_pbits = set(layout.values())

    layout_map = {}
    layout_map.update(layout)
    new_vbits = list(vbits - layout_vbits)
    new_pbits = list(pbits - layout_pbits)
    for i in range(len(new_vbits)):
        layout_map[new_vbits[i]] = new_pbits[i]
    # print(layout_map)

    return layout_map


def gen_layout(circ, backend, count=1e9):
    int_graph = qubit_interaction_graph(circ, count=count)
    logical_graph = nx.Graph()
    for (q1, q2), weight in int_graph.items():
        logical_graph.add_edge(q1, q2, weight=weight)

    physical_graph = nx.Graph()
    physical_graph.add_edges_from(backend.coupling_map)

    GM = nx.algorithms.isomorphism.GraphMatcher(physical_graph, logical_graph)
    if GM.is_isomorphic():
        for subgraph in GM.subgraph_isomorphisms_iter():
            print("Match found:", subgraph)

    starting_physical_node = find_start_point(backend.coupling_map)

    logical_ordered = generate_ordered_list_logical(logical_graph)
    physical_ordered = generate_ordered_list_physical(physical_graph, starting_physical_node)

    import random
    idx1, idx2 = random.sample(range(len(logical_ordered)), 2)
    logical_ordered[idx1], logical_ordered[idx2] = logical_ordered[idx2], logical_ordered[idx1]
    # import random
    # random.shuffle(logical_ordered)
    # print(logical_ordered)
    # Create mapping
    mapping = {}
    qubits = circ.qubits
    for i, logical_node in enumerate(logical_ordered):
        # mapping[logical_node] = physical_ordered[i]
        mapping[qubits[logical_node]] = physical_ordered[i]
        if qubits[logical_node]._index != logical_node:
            print("error")
    return mapping