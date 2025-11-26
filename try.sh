python - << 'EOF'
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

# 2 量子比特 + 自动创建 classical bits
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1]) 

backend = Aer.get_backend("aer_simulator")

qc_compiled = transpile(qc, backend)
job = backend.run(qc_compiled, shots=1000)
result = job.result()

counts = result.get_counts()
print(counts)
EOF

./analyze_qcbm.sh data/gen_qcbm_round1_scored.csv data/clean_kras_g12d.csv 50