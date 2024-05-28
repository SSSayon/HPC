import matplotlib.pyplot as plt

with open('am.graph', 'r') as file:
    lines = file.readlines()

ptr = list(map(int, lines[0].strip().split()))
idx = list(map(int, lines[1].strip().split()))

row_counts = [ptr[i+1] - ptr[i] for i in range(len(ptr)-1)]
M = 1000
# summarized_counts = [sum(row_counts[i:i+M]) for i in range(0, len(row_counts), M)]
summarized_counts = row_counts[M : M+1000]

with open('graph_structure.txt', 'w') as file:
    file.write(f'{summarized_counts}')

plt.figure(figsize=(10, 6))
plt.bar(range(len(summarized_counts)), summarized_counts, color='skyblue')
plt.xlabel(f'Chunk Index ({M} rows per chunk)')
plt.ylabel('Number of Elements')
plt.title(f'Number of Elements per {M} Rows in CSR Matrix')
plt.show()
