import numpy as np
import matplotlib.pyplot as plt

THETAS = {0.0: "Forward Euler", 0.5: "Crank-Nicolson", 1.0: "Backward Euler"}

def A_exact(p: float) -> float:
    return np.exp(-p)

def A(p, theta):
    """theta-rule one-step factor u^{n+1} / u^n, with p = a*dt."""

    return (1 - (1 - theta) * p) / (1 + theta * p)

def solver(I, a, T, dt, theta):
    """Solve u' = -a u, u(0) = I, on (0, T] with theta rule."""
    Nt = int(round(T / dt))
    t = np.linspace(0, Nt * dt, Nt + 1)
    u = np.zeros(Nt + 1)
    u[0] = I
    for n in range(Nt):
        u[n + 1] = (1 - (1 - theta) * a * dt) / (1 + theta * a * dt) * u[n]
    return u, t

def plot_experiments(I, a, T, dt_values, filename="growth_experiments.pdf"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    t_fine = np.linspace(0, T, 1001)

    for ax, dt in zip(axes.flat, dt_values):
        for theta, name in THETAS.items():
            u, t = solver(I, a, T, dt, theta)
            ax.plot(t, u, "o-", label=f"{name} (Theta={theta})")
        ax.plot(t_fine, I * np.exp(-a * t_fine), "k-", label="Exact")
        ax.set_xlabel("t")
        ax.set_ylabel("u")
        ax.set_title(f"dt = {dt}, a*dt = {a * dt}")

    axes.flat[0].legend()
    fig.suptitle(f"u' = -a u with a = {a}")
    fig.tight_layout(
    )
    fig.savefig(filename)

def plot_amplification(pmin=-3.0, ylim=(-10,15), filename="growth_amp.pdf"):
    p = np.linspace(pmin, 0, 2001)
    plt.figure(figsize=(8, 6))

    for theta, name in THETAS.items():
        with np.errstate(divide='ignore'):
            A_vals = A(p, theta)
        A_vals[np.abs(A_vals) > 10 * max(np.abs(ylim))] = np.nan
        A_vals[np.abs(np.diff(A_vals, prepend=A_vals[0])) > 5] = np.nan
        plt.plot(p, A_vals, label=f"{name} (Theta={theta})")
    
    plt.plot(p, A_exact(p), "k--", label="Exact A(p) = exp(-p)")
    plt.axhline(0, color="gray", lw=0.5)
    plt.axhline(1, color="gray", lw=0.5, ls=":")
    plt.ylim(*ylim)
    plt.xlabel("p = a*dt")
    plt.ylabel("A(p)")
    plt.title("Amplification factor for exp growth a * dt < 0")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)

if __name__ == "__main__":
    I, a, T = 1.0, -1.0, 6.0

    dt_values = [0.1, 0.5, 1.5, 3.0]

    print(f"a = {a}")
    for theta, name in THETAS.items():
        print(f"{name} (\nTheta={theta} ({name})")
        for dt in dt_values:
            p = a * dt
            print(f" dt = {dt:3}, a*dt  = {p:5}, A_exact(p) = {A_exact(p):8.4f}, A_theta(p) = {A(p, theta):8.4f}")

    plot_experiments(I, a, T, dt_values)
    plot_amplification()




