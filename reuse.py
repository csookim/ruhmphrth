from qiskit.circuit import Reset
from qiskit.converters import circuit_to_dag
from utils import preprocessing, qubit_interaction_graph

class ReuseHeuristic:
    def __init__(self):
        self.reuse_qubits = 0
        self.total_qubits = 0
        pass

    def run(self, circuit):
        qc = preprocessing(circuit)
        active_qubits = list(range(qc.num_qubits))
        reuse_pairs = self._find_reuse_pairs(qc, active_qubits)

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

            reuse_pairs = self._find_reuse_pairs(modified_qc, active_qubits)
            cur_qc = modified_qc.copy()
        
        self.reuse_qubits = i
        self.total_qubits = len(active_qubits)
        return cur_qc
        
    
    def _find_reuse_pairs(self, circuit, active_qubits):
        qiskit_dag = circuit_to_dag(circuit)
        custom_dag = self._my_custom_dag(circuit)

        reusable_pairs = []

        last_i = self._last_index_operation(circuit)
        first_i = self._first_index_operation(circuit)

        for i in active_qubits:
            if i not in last_i:
                continue
            last_op_index_i = last_i[i]
            for j in active_qubits:
                if i == j:
                    continue
                if j not in first_i:
                    continue
                first_op_index_j = first_i[j]

                if not self._share_same_gate(qiskit_dag, i, j) \
                    and not self._has_cycle(custom_dag, last_op_index_i, first_op_index_j):
                    reusable_pairs.append((i, j))

        return reusable_pairs
    
    def _share_same_gate(self, qiskit_dag, i, j):
        for node in qiskit_dag.topological_op_nodes():
            qubits = [qubit._index for qubit in node.qargs]
            if i in qubits and j in qubits:
                return True
        return False

    def _has_cycle(self, graph, i, j):
        if i < j:
            return False

        visited = set()
        stack = [j]
        cycle_detected = False
        while len(stack) > 0:
            node = stack.pop(0)
            visited.add(node)
            if node == i:
                cycle_detected = True
                break
            for neighbor in graph.get(node, []):
                if neighbor not in visited and neighbor not in stack and node <= i:
                    stack.append(neighbor)

        return cycle_detected
    
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
    
    def _last_index_operation(self, circuit):
        last_index = {}
        for i, gate in enumerate(circuit.data):
            for q in gate.qubits:
                last_index[q._index] = i
        return last_index


    def _first_index_operation(self, circuit):
        first_index = {}
        for i, gate in enumerate(circuit.data):
            for q in gate.qubits:
                if q._index not in first_index:
                    first_index[q._index] = i
        return first_index