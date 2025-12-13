import matplotlib.pyplot as plt
import numpy as np

# Number of pages processed
pages = np.arange(1, 122)

# Simulate gradual increase in tokens/sec
# Input tokens/sec goes from ~250 to ~2750
input_toks = np.linspace(250, 2750, len(pages))
# Output tokens/sec goes from ~60 to ~3075
output_toks = np.linspace(60, 3075, len(pages))

plt.figure(figsize=(12,6))
plt.plot(pages, input_toks, label='Input tokens/sec', color='skyblue', linewidth=2)
plt.plot(pages, output_toks, label='Output tokens/sec', color='orange', linewidth=2)

# Add markers at intervals for clarity
plt.scatter(pages[::10], input_toks[::10], color='blue')
plt.scatter(pages[::10], output_toks[::10], color='red')

# Labels and title
plt.title('Tokens/sec Processing Progress', fontsize=16, weight='bold')
plt.xlabel('Pages Processed', fontsize=14)
plt.ylabel('Tokens/sec', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)

# Optional: add a horizontal line at 3000 for reference
plt.axhline(y=3000, color='green', linestyle='--', alpha=0.5, label='Target 3k tokens/sec')

plt.tight_layout()
plt.show()