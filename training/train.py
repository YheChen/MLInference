import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main() -> None:
    rng = np.random.default_rng(0)

    n = 5000
    d = 10
    X = rng.normal(size=(n, d))
    w = rng.normal(size=(d,))
    logits = X @ w
    y = (logits > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    out_path = os.path.join("app", "model", "model.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)

    acc = model.score(X_test, y_test)
    print(f"Saved model to {out_path}. Test accuracy={acc:.3f}")


if __name__ == "__main__":
    main()
