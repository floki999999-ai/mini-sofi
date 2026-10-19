import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# MINI-SOFI 0.7
# STESSO MONDO DELLA 0.6
# CERVELLO PIU' GRANDE
# ==========================================

rng = np.random.default_rng()

# Carica lo STESSO mondo.csv della 0.6
data = np.loadtxt(
    "mondo.csv",
    delimiter=",",
    skiprows=1
)

X = data[:, :2]
y = data[:, 2].reshape(-1, 1)

# Mescoliamo i dati
indices = rng.permutation(len(X))
X = X[indices]
y = y[indices]

# Stessa divisione della 0.6
X_train = X[:7000]
y_train = y[:7000]

X_val = X[7000:8500]
y_val = y[7000:8500]

X_test = X[8500:]
y_test = y[8500:]


# ==========================================
# FUNZIONI
# ==========================================

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(output):
    return output * (1 - output)


# ==========================================
# CERVELLO MINI-SOFI 0.7
#
# 2 -> 64 -> 64 -> 32 -> 1
# ==========================================

W1 = rng.uniform(-1, 1, (2, 64))
b1 = np.zeros((1, 64))

W2 = rng.uniform(-1, 1, (64, 64))
b2 = np.zeros((1, 64))

W3 = rng.uniform(-1, 1, (64, 32))
b3 = np.zeros((1, 32))

W4 = rng.uniform(-1, 1, (32, 1))
b4 = np.zeros((1, 1))

learning_rate = 0.8
epochs = 30000


print("\n==============================")
print("       MINI-SOFI 0.7")
print("==============================")

print("\nCervello: 2 -> 64 -> 64 -> 32 -> 1")
print("Studio:      7000 esempi")
print("Validazione: 1500 esempi")
print("Esame:       1500 esempi")
print("\nStesso mondo della Mini-Sofi 0.6.")
print("Vediamo se un cervello piu' grande capisce di piu'...\n")


# ==========================================
# ADDESTRAMENTO
# ==========================================

for epoch in range(epochs + 1):

    # ---------- FORWARD ----------

    h1 = sigmoid(X_train @ W1 + b1)

    h2 = sigmoid(h1 @ W2 + b2)

    h3 = sigmoid(h2 @ W3 + b3)

    output = sigmoid(h3 @ W4 + b4)


    # ---------- ERRORE ----------

    error = y_train - output
    loss = np.mean(error ** 2)


    # ---------- BACKPROPAGATION ----------

    delta4 = (
        error *
        sigmoid_derivative(output)
    )

    error3 = delta4 @ W4.T

    delta3 = (
        error3 *
        sigmoid_derivative(h3)
    )

    error2 = delta3 @ W3.T

    delta2 = (
        error2 *
        sigmoid_derivative(h2)
    )

    error1 = delta2 @ W2.T

    delta1 = (
        error1 *
        sigmoid_derivative(h1)
    )


    # ---------- AGGIORNAMENTO PESI ----------

    W4 += (
        learning_rate *
        h3.T @ delta4 /
        len(X_train)
    )

    b4 += (
        learning_rate *
        np.mean(delta4, axis=0, keepdims=True)
    )


    W3 += (
        learning_rate *
        h2.T @ delta3 /
        len(X_train)
    )

    b3 += (
        learning_rate *
        np.mean(delta3, axis=0, keepdims=True)
    )


    W2 += (
        learning_rate *
        h1.T @ delta2 /
        len(X_train)
    )

    b2 += (
        learning_rate *
        np.mean(delta2, axis=0, keepdims=True)
    )


    W1 += (
        learning_rate *
        X_train.T @ delta1 /
        len(X_train)
    )

    b1 += (
        learning_rate *
        np.mean(delta1, axis=0, keepdims=True)
    )


    # ---------- VALIDAZIONE ----------

    if epoch % 1000 == 0:

        vh1 = sigmoid(X_val @ W1 + b1)
        vh2 = sigmoid(vh1 @ W2 + b2)
        vh3 = sigmoid(vh2 @ W3 + b3)

        vout = sigmoid(
            vh3 @ W4 + b4
        )

        vpred = (
            vout >= 0.5
        ).astype(int)

        val_accuracy = np.mean(
            vpred == y_val
        ) * 100

        print(
            f"Epoca {epoch:5d} | "
            f"errore {loss:.6f} | "
            f"validazione {val_accuracy:.2f}%"
        )


# ==========================================
# ESAME FINALE
# ==========================================

th1 = sigmoid(X_test @ W1 + b1)
th2 = sigmoid(th1 @ W2 + b2)
th3 = sigmoid(th2 @ W3 + b3)

tout = sigmoid(
    th3 @ W4 + b4
)

test_prediction = (
    tout >= 0.5
).astype(int)

correct = np.sum(
    test_prediction == y_test
)

accuracy = (
    correct /
    len(y_test) *
    100
)


print("\n==============================")
print("        ESAME FINALE")
print("==============================")

print(
    f"\nCorrette: {correct}/{len(y_test)}"
)

print(
    f"Accuratezza: {accuracy:.2f}%"
)


# ==========================================
# MAPPA DEL MONDO SECONDO MINI-SOFI
# ==========================================

resolution = 250

axis = np.linspace(
    0,
    1,
    resolution
)

xx, yy = np.meshgrid(
    axis,
    axis
)

grid = np.c_[
    xx.ravel(),
    yy.ravel()
]

gh1 = sigmoid(
    grid @ W1 + b1
)

gh2 = sigmoid(
    gh1 @ W2 + b2
)

gh3 = sigmoid(
    gh2 @ W3 + b3
)

gout = sigmoid(
    gh3 @ W4 + b4
)

zz = gout.reshape(
    xx.shape
)


plt.figure(
    figsize=(8, 7)
)

mappa = plt.contourf(
    xx,
    yy,
    zz,
    levels=50,
    vmin=0,
    vmax=1
)

plt.colorbar(
    mappa,
    label="Risposta Mini-Sofi"
)

# Confine deciso dalla rete

plt.contour(
    xx,
    yy,
    zz,
    levels=[0.5],
    linewidths=2
)

# Alcune domande dell'esame

plt.scatter(
    X_test[:300, 0],
    X_test[:300, 1],
    c=y_test[:300, 0],
    s=10,
    alpha=0.5
)

plt.xlabel("Input 1")
plt.ylabel("Input 2")

plt.title(
    "Mini-Sofi 0.7\n"
    f"Cervello 64-64-32 | "
    f"Esame: {accuracy:.2f}%"
)

plt.tight_layout()

plt.show()