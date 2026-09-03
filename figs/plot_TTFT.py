import matplotlib.pyplot as plt

concurrency = [1,2,4,6,8]
TTFT_P50 = []
TTFT_P95 = []
TTFT_mean = []

fig, axis = plt.subplots(figsize=(6, 4))
axis.patch.set_facecolor('white')

axis.plot(concurrency, TTFT_mean, marker='o', label='TTFT mean', color='blue')

axis.legend()
axis.grid(True, alpha=0.3) 
plt.tight_layout()
plt.show()