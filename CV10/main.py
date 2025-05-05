import numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path


def logistic_next(x, a):  # x_{n+1} = a·x_n(1‑x_n)
    return a * x * (1.0 - x)


def make_bifurcation(a_vals, n_iter=1200, n_skip=200):
    xs, aa = [], []
    for a in a_vals:
        x = 0.5  # same start for every a
        for i in range(n_iter):
            x = logistic_next(x, a)
            if i >= n_skip:  # drop transient
                xs.append(x)
                aa.append(a)
    return np.asarray(aa), np.asarray(xs)


class Net(nn.Module):  # 2‑→128‑128‑64‑32‑1
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Dropout(0.03),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_nn(samples=200_000, epochs=50, batch=512, lr=1e-3):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    a = torch.rand(samples, 1) * 4
    x = torch.rand(samples, 1)
    y = logistic_next(x, a)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.cat([a, x], 1), y),
        batch_size=batch,
        shuffle=True,
    )

    net = Net().to(dev)
    opt = optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()

    for ep in range(1, epochs + 1):
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.detach().item()
        print(f"ep{ep:02d}  mse={tot/len(loader):.3e}")
    return net.cpu()


def bifurcation_via_nn(net, a_vals, n_iter=600, n_skip=20, stride=5):
    xs, aa = [], []
    net.eval()
    with torch.no_grad():
        for a in a_vals:
            x = 0.5
            for i in range(n_iter):
                x = float(net(torch.tensor([[a, x]], dtype=torch.float32))[0, 0])
                if i >= n_skip and (i - n_skip) % stride == 0:  # keep 1/stride points
                    xs.append(x)
                    aa.append(a)
    return np.asarray(aa), np.asarray(xs)


def plot_both(a_exact, x_exact, a_pred, x_pred):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ax[0].scatter(a_exact, x_exact, s=0.1, c="tab:blue")
    ax[0].set_title("Exact")
    ax[1].scatter(a_pred, x_pred, s=0.1, c="tab:orange")
    ax[1].set_title("NN")
    for a in ax:
        a.set(xlim=(0, 4), ylim=(0, 1), xlabel="a")
    ax[0].set_ylabel("x")
    plt.tight_layout()
    plt.show()


MODEL_PATH = Path("CV10/weights.pth")
CSV_PATH = Path("CV10/predicted_points.csv")


def main():
    a_grid = np.linspace(0, 4, 1500)
    a_exact, x_exact = make_bifurcation(a_grid)

    retrain = "y" if not MODEL_PATH.exists() else input("Retrain model? (y/N) ").lower()
    if retrain == "y":
        net = train_nn()
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), MODEL_PATH)
    else:
        net = Net()
        net.load_state_dict(torch.load(MODEL_PATH))
        net.eval()

    use_csv = (
        CSV_PATH.exists() and input("Load cached prediction? (Y/n) ").lower() != "n"
    )
    if use_csv:
        data = np.loadtxt(CSV_PATH, delimiter=",")
        a_pred, x_pred = data[:, 0], data[:, 1]
    else:
        a_pred, x_pred = bifurcation_via_nn(net, a_grid)
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            CSV_PATH, np.column_stack([a_pred, x_pred]), delimiter=",", fmt="%.6f"
        )

    plot_both(a_exact, x_exact, a_pred, x_pred)


if __name__ == "__main__":
    main()
