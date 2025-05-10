from qiskit_ibm_runtime.fake_provider import FakeTokyo, FakeMumbaiV2
from qiskit.providers.fake_provider import GenericBackendV2
from utils import preprocessing, gen_layout, count_cx
from qiskit.transpiler import Layout, PassManager
from qiskit.transpiler.passes import SetLayout, ApplyLayout, FullAncillaAllocation, EnlargeWithAncilla
from qiskit_sabre import SabreSwap_old
from reuse import ReuseHeuristic
from qiskit import transpile
import random

def FakeTokyoV2():
    cmap_tokyo = [(0, 1), (0, 5), (1, 2), (1, 6), (1, 7), 
                (2, 3), (2, 6), (2, 7), (3, 4), (3, 8), 
                (3, 9), (4, 8), (4, 9), (5, 6), (5, 10), 
                (5, 11), (6, 7), (6, 10), (6, 11), (7, 8), 
                (7, 12), (7, 13), (8, 9), (8, 12), (8, 13), 
                (9, 14), (10, 11), (10, 15), (11, 12), (11, 16), 
                (11, 17), (12, 13), (12, 16), (12, 17), (13, 14), 
                (13, 18), (13, 19), (14, 18), (14, 19), (15, 16), (16, 17), (17, 18), (18, 19)]
    nq = FakeTokyo().configuration().num_qubits
    backend = GenericBackendV2(num_qubits=nq, coupling_map=cmap_tokyo, noise_info=False)
    return backend

def sabre(circ, ruh=False, sabre_old=False, backend=None):
    if backend is None:
        backend = FakeMumbaiV2()
    coup = backend.coupling_map
    circ = preprocessing(circ)
    if ruh:
        ruh = ReuseHeuristic()
        circ = ruh.run(circ)
    t_circ_qiskit = transpile(circ, backend, layout_method='sabre', routing_method='sabre', optimization_level=0)
    if sabre_old:
        layout = t_circ_qiskit.layout.initial_layout.get_virtual_bits()
        t_circ_qiskit = transpile_new(circ, layout, coup)
    return t_circ_qiskit

def transpile_new(circ, layout:dict, coup, ext_size=30):
    pass_s = PassManager()
    pass_s.append(SetLayout(Layout(layout)))
    pass_s.append(FullAncillaAllocation(coup))
    pass_s.append(EnlargeWithAncilla())
    pass_s.append(ApplyLayout())
    pass_s.append(SabreSwap_old(coup, "lookahead", bridge=True, extended_set_size=ext_size))
    transpiled_circ = pass_s.run(circ)
    return transpiled_circ

def slice_circuit(circ, count):
    new_circ = circ.copy_empty_like()
    rev_circ = circ.copy_empty_like()
    c = 0
    for gate in circ:
        if c == count:
            break
        if len(gate.qubits) == 2:
            new_circ.append(gate)
            rev_circ.append(gate)
            c += 1

    for gate in rev_circ[::-1]:
        new_circ.append(gate)
    return new_circ

def reverse_mapping(sliced_circ, ext_size, coup, backend, layout:dict=None, sabre_old=False, count=10):
    # sliced_circ = slice_circuit(circ, count)

    if not sabre_old:
        # fw1 = transpile(for_circ, backend, layout_method='sabre', routing_method='sabre', optimization_level=0, initial_layout=layout)
        # fw1_fin_layout: dict = fw1.layout.final_virtual_layout(True).get_virtual_bits()
        # rev1 = transpile(rev_circ, backend, layout_method='sabre', routing_method='sabre', optimization_level=0, initial_layout=fw1_fin_layout)
        # rev1_fin_layout: dict = rev1.layout.final_virtual_layout(True).get_virtual_bits()
        # return rev1_fin_layout
        t_circ = transpile(sliced_circ, backend, layout_method='sabre', routing_method='sabre', optimization_level=0, initial_layout=layout)
        return t_circ.layout.final_virtual_layout(True).get_virtual_bits()
    else:
        # fw1_old = transpile_new(for_circ, layout, coup, ext_size)
        # fw1_old_fin_layout: dict = fw1_old.layout.final_virtual_layout(True).get_virtual_bits()
        # rev1_old = transpile_new(rev_circ, fw1_old_fin_layout, coup, ext_size)
        # rev1_old_fin_layout: dict = rev1_old.layout.final_virtual_layout(True).get_virtual_bits()
        # return rev1_old_fin_layout
        t_circ = transpile_new(sliced_circ, layout, coup, ext_size)
        return t_circ.layout.final_virtual_layout(True).get_virtual_bits()
    
def best_mapping(circ, backend, ext_size, count, max_eval):
    best_cxs = 1e9
    best_layout = None
    for _ in range(max_eval):
        layout = gen_layout(circ, backend, count)
        layout = reverse_mapping(circ, ext_size, backend.coupling_map, backend, layout, count=count)
        l1, l2 = random.sample(list(layout.keys()), 2)
        layout[l1], layout[l2] = layout[l2], layout[l1]
        t_circ = transpile_new(circ, layout, backend.coupling_map, ext_size=ext_size)
        cx_count = count_cx(t_circ)
        if cx_count < best_cxs:
            best_cxs = cx_count
            best_layout = layout
    return best_layout

def best_mapping2(circ, backend, ext_size, count, max_eval):
    best_layout = gen_layout(circ, backend, count)
    t_circ = transpile_new(circ, best_layout, backend.coupling_map, ext_size=ext_size)
    best_cxs = count_cx(t_circ)

    circ_part = circ.copy_empty_like()
    c = 0
    for gate in circ:
        if len(gate.qubits) == 2:
            circ_part.append(gate.operation, gate.qubits)
            c += 1
            if c > count:
                break

    for _ in range(max_eval):
        layout = best_layout.copy()
        l1, l2 = random.sample(list(layout.keys()), 2)
        layout[l1], layout[l2] = layout[l2], layout[l1]

        t_circ = transpile_new(circ_part, layout, backend.coupling_map, ext_size=ext_size)
        cx_count = count_cx(t_circ)
        if cx_count < best_cxs:
            best_cxs = cx_count
            best_layout = layout
    return best_layout

def random_l1_l2(circ, layout):
    first_gate = circ[0]
    fg_l1 = first_gate.qubits[0]
    fg_l2 = first_gate.qubits[1]
    candidate_l = list(layout.keys())
    if len(list(layout.keys())) > 5:
        candidate_l.remove(fg_l1)
        candidate_l.remove(fg_l2)
    l1, l2 = random.sample(candidate_l, 2)
    return l1, l2
    

def best_mapping3(circ, backend, ext_size, count, max_eval):
    sliced_circ = slice_circuit(circ, count)

    best_layout = gen_layout(circ, backend, count)
    best_layout = reverse_mapping(sliced_circ, ext_size, backend.coupling_map, backend, best_layout, count=count)
    t_circ = transpile_new(circ, best_layout, backend.coupling_map, ext_size=ext_size)
    best_cxs = count_cx(t_circ)
    
    for _ in range(max_eval):
        # layout = gen_layout(circ, backend, count)
        layout = best_layout.copy()
        l1, l2 = random_l1_l2(sliced_circ, layout)
        layout[l1], layout[l2] = layout[l2], layout[l1]
        # layout = reverse_mapping(sliced_circ, ext_size, backend.coupling_map, backend, best_layout, count=count)

        t_circ = transpile_new(circ, layout, backend.coupling_map, ext_size=ext_size)
        cx_count = count_cx(t_circ)
        if cx_count < best_cxs:
            best_cxs = cx_count
            best_layout = layout
    return best_layout

def best_mapping4(circ, backend, ext_size, count, max_eval):
    sliced_circ = slice_circuit(circ, count)

    best_layout = gen_layout(circ, backend, count)
    best_layout = reverse_mapping(sliced_circ, ext_size, backend.coupling_map, backend, best_layout, count=count)
    t_circ = transpile_new(circ, best_layout, backend.coupling_map, ext_size=ext_size)
    best_cxs = count_cx(t_circ)
    
    for _ in range(max_eval):
        layout = best_layout.copy()
        l1, l2 = random.sample(list(layout.keys()), 2)
        layout[l1], layout[l2] = layout[l2], layout[l1]

        t_circ = transpile_new(circ, layout, backend.coupling_map, ext_size=ext_size)
        cx_count = count_cx(t_circ)
        if cx_count < best_cxs:
            best_cxs = cx_count
            best_layout = layout
    return best_layout