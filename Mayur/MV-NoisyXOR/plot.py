import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    dir = "./Mayur/MV-NoisyXOR/results"
    noises = [0.01, 0.05, 0.1, 0.2]
    num_values = [50, 100, 200, 400, 800]

    bests = np.zeros((len(noises), len(num_values)))

    fig, axs = plt.subplots(
        len(noises),
        len(num_values),
        figsize=(15, 10),
        sharex=True,
        sharey=True,
    )
    fig.suptitle("Noisy MultivalueXOR")
    fig.supxlabel("Number of values")
    fig.supylabel("Percentage of Noise")
    for i, noise in enumerate(noises):
        for j, num_value in enumerate(num_values):
            df = pd.read_csv(f"{dir}/noisy_xor_{num_value}_{noise}.csv")
            df.plot(ax=axs[3 - i, j])
            axs[3 - i, j].set_xlabel(f"Epochs(num_values = {num_value})")
            axs[3 - i, j].set_ylabel(f"Accuracy(noise = {noise})")

            best = df.max(axis=0).min()

            bests[i, j] = best

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Trial",
        loc="center right",
    )

    for ax in axs.ravel():
        ax.get_legend().remove()

    fig_heat, ax_heat = plt.subplots(layout="compressed")
    sns.heatmap(
        bests,
        annot=True,
        cbar=True,
        ax=ax_heat,
        yticklabels=[f"{x}" for x in noises],
        xticklabels=[f"{x}" for x in num_values],
    )
    ax_heat.invert_yaxis()
    ax_heat.set_xlabel("Number of values")
    ax_heat.set_ylabel("Percentage of Noise")
    ax_heat.set_title("Heatmap of Noise vs Number of values")

    fig_heat.savefig(f"{dir}/heatmap.png")
    fig.savefig(f"{dir}/line-plot.png")
