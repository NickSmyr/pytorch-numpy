from matplotlib import pyplot as plt


def plot_loss_cost_accuracy_on_val_train(x):
    fig, axs = plt.subplots(1, 3)

    for target in ["val", "train"]:
        for i, metric in enumerate(["accuracy", "loss", "cost"]):
            target_metric_values = x[target][metric]
            axs[i].plot(target_metric_values, label=f"{target} {metric}")
            axs[i].set_title(f"{metric} on val/train")
            axs[i].legend(loc="lower left")

    plt.savefig("val-train-metrics.pdf")