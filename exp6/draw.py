import matplotlib.pyplot as plt

strides = [1, 2, 4, 8]
data = [530.105, 182.482, 91.9984, 46.2854]

plt.figure(figsize=(10, 6))
plt.plot(strides, data, marker='o')
plt.xlabel('Stride')
plt.ylabel('Bandwidth')
plt.title('Bandwidth for different Strides (gmem)')
plt.grid(True)
plt.show()

# strides = [1, 2, 4, 8, 16, 32]
# bitwidth_2 = [4303.03, 4303.49, 2156.9, 832.619, 428.382, 215.588]
# bitwidth_4 = [8605.54, 4318.46, 2026.51, 1019.25, 509.914, 252.267]
# bitwidth_8 = [8657.77, 4339.3, 2173.55, 1087.65, 544.069, 544.07]

# plt.figure(figsize=(10, 6))
# plt.plot(strides, bitwidth_2, marker='o', label='Bitwidth=2')
# plt.plot(strides, bitwidth_4, marker='o', label='Bitwidth=4')
# plt.plot(strides, bitwidth_8, marker='o', label='Bitwidth=8')
# plt.xlabel('Stride')
# plt.ylabel('Bandwidth')
# plt.title('Bandwidth vs. Stride for different Bitwidths (smem)')
# plt.legend()
# plt.grid(True)
# plt.show()
