from src.utils.result import load_lines_in_range_jsonl
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
import matplotlib.pyplot as plt


results = load_lines_in_range_jsonl("p_stability", -1)["results"]

# Plot the bounds
exact_p = results["Exact p"]
lower_bound_p = results["Lower Bound p"]
better_upper_bound_p = results["Better Upper Bound p"]
better_lower_bound_stability = results["Better Lower Bound Stability"]
upper_bound_stability = results["Upper Bound Stability"]

# Create a square plot
plt.figure(figsize=(18, 18))  # Set the figure size to 8x8 inches (square)

# P is X-axis, stability is Y-axis
plt.plot(
    exact_p["p"], exact_p["stability"], label="Exact p", linewidth=2, linestyle="-"
)
plt.plot(
    lower_bound_p["p"],
    lower_bound_p["stability"],
    label="Lower Bound p",
    linewidth=2,
    linestyle="--",
)
plt.plot(
    better_upper_bound_p["p"],
    better_upper_bound_p["stability"],
    label="Better Upper Bound p",
    linewidth=2,
    linestyle=":",
)
plt.plot(
    better_lower_bound_stability["p"],
    better_lower_bound_stability["stability"],
    label="Better Lower Bound Stability",
    linewidth=2,
    linestyle="-.",
)
plt.plot(
    upper_bound_stability["p"],
    upper_bound_stability["stability"],
    label="Upper Bound Stability",
    linewidth=2,
    linestyle=(0, (3, 1, 1, 1)),
)

plt.xlabel("p")
plt.ylabel("Stability")
plt.title("Bounds for p and Stability")
plt.legend()

# Save the plot
plt.savefig("results/figures/bounds.png")
plt.show()

a = 1
