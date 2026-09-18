"""Held-out CPU experiment on sklearn's bundled digits, not MNIST/EMNIST."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from digit_audio import CompactUNet, target, write_audio


def metrics(pred, truth):
    flat, ref = pred.reshape(len(pred), -1), truth.reshape(len(truth), -1)
    cos = np.sum(flat * ref, axis=1) / np.maximum(np.linalg.norm(flat, axis=1)*np.linalg.norm(ref, axis=1), 1e-12)
    return {"mse": float(np.mean((pred-truth)**2)), "mean_cosine_similarity": float(np.mean(cos))}


def run(output, steps, seed):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    data = load_digits()
    train, held = train_test_split(np.arange(len(data.target)), train_size=600, test_size=200,
                                   stratify=data.target, random_state=seed)
    images = data.images.astype(np.float32)/16
    x = torch.from_numpy(images[:, None])
    targets = np.stack([target(images[i], data.target[i]) for i in np.r_[train, held]])
    y_train, y_held = targets[:len(train)], targets[len(train):]
    model = CompactUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = []
    train_tensor = torch.from_numpy(y_train[:, None])
    model.train()
    for step in range(steps):
        batch = rng.choice(len(train), 16, replace=False)
        prediction = model(x[train[batch]])
        truth = train_tensor[batch]
        loss = torch.mean((prediction-truth)**2 * (1 + 8*truth))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step % 10 == 0 or step == steps-1:
            history.append({"step": step+1, "weighted_train_mse": float(loss.detach())})
    model.eval()
    with torch.no_grad():
        predictions = np.concatenate([model(chunk).numpy()[:, 0] for chunk in x[held].split(16)])
    classifier = LogisticRegression(C=10, max_iter=1000, random_state=seed)
    classifier.fit(images[train].reshape(len(train), -1), data.target[train])
    labels = classifier.predict(images[held].reshape(len(held), -1))
    hybrid = np.stack([target(images[i], label) for i, label in zip(held, labels)])
    mean = np.broadcast_to(y_train.mean(axis=0), y_held.shape)
    report = {"dataset": "sklearn bundled 8x8 optical digits, NOT MNIST/EMNIST", "seed": seed,
              "train_indices": train.tolist(), "held_out_indices": held.tolist(), "updates": steps,
              "parameters": sum(p.numel() for p in model.parameters()),
              "target_sha256": hashlib.sha256(targets.tobytes()).hexdigest(),
              "held_out": {"train_mean": metrics(mean, y_held), "compact_unet": metrics(predictions, y_held),
                           "classify_then_synthesize": metrics(hybrid, y_held)},
              "classifier_accuracy": float(np.mean(labels == data.target[held])), "history": history,
              "caveat": "Targets are deterministic procedural labels. Hybrid uses predicted labels only; target uses true labels. Not a natural music generation benchmark."}
    checkpoint = {"state_dict": model.state_dict(), "seed": seed, "steps": steps, "output_shape": [129, 64], "channels": 8}
    torch.save(checkpoint, output / "compact-unet.pt")
    restored = CompactUNet()
    restored.load_state_dict(torch.load(output / "compact-unet.pt", weights_only=True)["state_dict"])
    restored.eval()
    with torch.no_grad():
        report["checkpoint_max_error"] = float(torch.max(torch.abs(restored(x[held[:1]]) - model(x[held[:1]]))))
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(3, 4, figsize=(11, 7), layout="constrained")
    for row, index in enumerate((0, 1, 2)):
        for col, (title, array) in enumerate([(f"Held-out digit {data.target[held[index]]}", images[held[index]]),
                                            ("Procedural target", y_held[index]), ("Compact U-Net", predictions[index]),
                                            (f"Classifier + DSP (label {labels[index]})", hybrid[index])]):
            axes[row, col].imshow(array, origin="lower" if col else "upper", aspect="auto", cmap="magma" if col else "gray", vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        for name, array in [("target", y_held[index]), ("unet", predictions[index]), ("hybrid", hybrid[index])]:
            write_audio(output / f"digit-{data.target[held[index]]}-{index}-{name}.wav", array, np.linspace(0, 2048, 129))
    fig.savefig(output / "comparison.png", dpi=160)
    print(json.dumps({"held_out": report["held_out"], "classifier_accuracy": report["classifier_accuracy"],
                      "checkpoint_max_error": report["checkpoint_max_error"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/cpu"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    run(args.output, args.steps, args.seed)
