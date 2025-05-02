from qiskit.circuit import Reset
from qiskit.converters import circuit_to_dag
from utils import preprocessing, qubit_interaction_graph    

class ReuseHeuristic:
    def __init__(self):
        self.reuse_qubits = 0
        self.total_qubits = 0

    def run(self, circuit):
        qc = preprocessing(circuit)
        active_qubits = list(range(qc.num_qubits))
        dependent_qubits, reuse_pairs = self._find_reuse_pairs(qc)

        i = 0
        cur_qc = qc.copy()
        chain = []

        while len(reuse_pairs) > 0 and i < len(qc.qubits) - 1:
            best = 0
            best_pair = reuse_pairs[0]
            for pair in reuse_pairs:
                fr = pair[1]
                to = pair[0]

                fr_edge_sum = 0
                to_edge_sum = 0
                graph = qubit_interaction_graph(cur_qc)
                for k, v in graph.items():
                    if fr in k:
                        fr_edge_sum += v
                    if to in k:
                        to_edge_sum += v

                if fr_edge_sum + to_edge_sum > best:
                    best = fr_edge_sum + to_edge_sum
                    best_pair = pair

            chain.append((best_pair[0], best_pair[1]))
            i += 1

            modified_qc = self._modify_circuit(cur_qc,best_pair)
            active_qubits.remove(best_pair[1])

            reuse_pairs = self._update_reuse_pairs(dependent_qubits, best_pair, reuse_pairs)
            cur_qc = modified_qc.copy()
        
        self.reuse_qubits = i
        self.total_qubits = len(active_qubits)
        return cur_qc
    
    def _share_same_gate(self, qiskit_dag, i, j):
        for node in qiskit_dag.topological_op_nodes():
            qubits = [qubit._index for qubit in node.qargs]
            if i in qubits and j in qubits:
                return True
        return False
    
    def _find_reuse_pairs(self, circuit):
        nqubits = circuit.num_qubits
        dep = [set([i]) for i in range(nqubits)]
        for gate in circuit[::-1]:
            if len(gate.qubits) == 2:
                q0 = gate.qubits[0]._index
                q1 = gate.qubits[1]._index
                dep[q0] |= dep[q1]
                dep[q1] |= dep[q0]

        reuse_pairs = []
        for i in range(nqubits):
            available = list(set(list(range(nqubits))) - dep[i])
            for j in available:
                reuse_pairs.append((j, i))
        return dep, reuse_pairs
    
    def _update_reuse_pairs(self, dependent_qubits, reuse_pair, reuse_pairs):
        c = reuse_pair[0]
        d = reuse_pair[1]

        for i, v in enumerate(dependent_qubits):
            if c in v:
                dependent_qubits[i] |= dependent_qubits[d]
            if d in v:
                dependent_qubits[i].add(c)
        dependent_qubits[c] |= dependent_qubits[d]
        dependent_qubits[d] |= dependent_qubits[c]
        new_reuse_pairs = []
        for pair in reuse_pairs:
            if pair[0] in dependent_qubits[pair[1]]:
                continue
            if d in pair:
                continue
            new_reuse_pairs.append(pair)

        return new_reuse_pairs
    
    def _modify_circuit(self, circuit, pair):
        i, j = pair

        operations = []
        check_list = []
        get_list = []
        visited = []
        for index, (inst, qargs, cargs) in enumerate(circuit.data):
            operations.append((inst, qargs, cargs))
            visited.append(index)

            if any(circuit.find_bit(q).index == i for q in qargs):
                check_list.append(index)
            if any(circuit.find_bit(q).index == j for q in qargs):
                get_list.append(index)

        #generate dag, and reverse it to form dependency lists
        forwards_adjecencies = self._my_custom_dag(circuit)
        dependencies = [[] for _ in range(len(operations))]
        for a, adj in forwards_adjecencies.items():
            for b in adj:
                dependencies[b].append(a)


        new_circuit = circuit.copy_empty_like()

        # Add all operations to the new circuit that do not depend on j or its descendants
        for index, (inst, qargs, cargs) in enumerate(operations):
            #condition 1 if a dependency has not been processed, condition 2 is if it contains j
            if any(n in visited for n in dependencies[index]) or index in get_list:
                continue
            new_circuit.append(inst, qargs, cargs)
            visited.remove(index)

        #as i should be done, we can do this. If i is not done, something went wrong with i and j
        new_circuit.append(Reset(), [i], [])

        # Process remaining operations, replacing qubit j with qubit i
        for index, (inst, qargs, cargs) in enumerate(operations):
            # print(operations[index])
            if  index in get_list:
                new_qargs = [new_circuit.qubits[i] if circuit.find_bit(q).index == j else q for q in qargs]
                new_circuit.append(inst, new_qargs, cargs)
                visited.remove(index)
            if index in visited:
                new_circuit.append(inst, qargs, cargs)
                visited.remove(index)
        if len(visited) != 0:
            print("something went wrong")
            print(visited)
        return new_circuit
    
    def _my_custom_dag(self, circuit):
        dag, vals = {}, {}
        vals.setdefault(None)
        for index, gate in enumerate(circuit.data):
            for q in gate.qubits:
                bit = circuit.find_bit(q).index
                if vals.get(bit) is not None:
                    if dag.get(vals[bit], None) is None:
                        dag[vals[bit]] = []
                    dag[vals[bit]].append(index)
                vals[bit] = index
        return dag