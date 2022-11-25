import json
import numpy as np
import matplotlib.pyplot as plt
from utils import label2id, id2label
from pathlib import Path
from sklearn import metrics
from scipy.interpolate import CubicSpline
import seaborn as sns

def plot_and_save_cm(cm, base_out):
    labels = list(reversed(label2id))
    plt.figure(base_out)
    ax = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='BuPu')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.plot()
    plt.savefig(f"plots/cm_{base_out}.pdf")
    
def plot_train_val_loss(log_history, base_out="test"):
    plt.figure(base_out)
    weirds = [point for point in log_history if "loss" not in point]
    hist_w_eval_loss = [point for point in log_history if "eval_loss" in point]
    eval_steps, eval_loss = list(zip(*([(point["step"], point["eval_loss"])
                                        for point in hist_w_eval_loss])))
    train_steps, train_loss = list(zip(*([(point["step"], point["loss"]) for point in log_history if "loss" in point])))
    xs = np.linspace(0,max(train_steps), 200)
    cs_eval = CubicSpline(eval_steps, eval_loss)
    cs_train = CubicSpline(train_steps, train_loss)
    fig, ax = plt.subplots()
    
    #plt.plot(eval_steps, eval_loss, c="purple", alpha=.3, label="Val loss")
    plt.plot(xs, cs_eval(xs), c="purple", label="Val loss")
    #plt.plot(train_steps, train_loss, c="blue", alpha=.3, label="Train loss")
    plt.plot(xs, cs_train(xs), c="blue", label="Train loss")
    plt.ylabel("Loss")
    plt.xlabel("Step")
    ax.legend()
    plt.savefig(f"plots/loss_{base_out}.pdf")
    
def print_and_plot_results(results, base_out):
    predictions_ids = np.argmax(results["logits"],axis=-1)
    predictions = np.array([id2label[label_id] for label_id in predictions_ids])
    references = np.array([id2label[label_id] for label_id in results["label_ids"]])
    report_matrix = metrics.classification_report(references, predictions, zero_division=0)
    cm = metrics.confusion_matrix(references, predictions)
    print(base_out)
    print(report_matrix)
    plot_and_save_cm(cm, base_out)
    return report_matrix, cm
    
if __name__ == "__main__":
    print("Loading Pretrained results")
    pre_path = "pretrained_eval_results.json"
    with open(pre_path, "r") as pretrained_file:
        pre_results = json.loads(json.load(pretrained_file))
    
    fine_paths = list(Path("finetuned_evals/").glob("eval_finetune_*.json"))
    
    # Plot training and validation loss, as well as classification and ConfMatrixes
    print("Handle finetuned results")
    for ft_path in fine_paths:
        print("Loading", ft_path)
        with open(ft_path, "r") as pretrained_file:
            ft_results =json.loads(json.load(pretrained_file))
        print(ft_path)
        base_out = "_".join(str(ft_path).split("_")[2:7])
        plot_train_val_loss(ft_results["log_history"], base_out)
        print_and_plot_results(ft_results, base_out)
    
    print("Handle pretrained results")
    # Calculate and plot reports for pre_trained models
    for model, model_ds in pre_results.items():
        for field, lang_ds in model_ds.items():
            for ds_name, ds_results in lang_ds.items():
                base_out = f"{model.split('/')[-1]}_{field}_{ds_name}"
                print_and_plot_results(ds_results, base_out)

                
    # Calculate and plot reports for finetuned models
    

        
    
    
