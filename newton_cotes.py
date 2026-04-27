"""
Newton-Cotes 数値積分

閉じたニュートン・コーツ公式による数値積分の実装。
  - 台形則 (n=1)
  - Simpson 1/3 則 (n=2)
  - Simpson 3/8 則 (n=3)
  - Boole 則 (n=4)
各単区間ルールと、区間を N 等分して適用する複合則を含む。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # 画面なし環境でも PNG 保存できるようにする
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================
# 単区間ルール
# ============================================================

def trapezoid(f, a: float, b: float) -> float:
    """台形則: [a, b] を 1 区間として積分（2 点）。
    次数 1 以下の多項式に対して厳密。
    誤差: -(b-a)^3/12 * f''(xi)
    """
    h = b - a
    return h / 2 * (f(a) + f(b))


def simpson13(f, a: float, b: float) -> float:
    """Simpson 1/3 則: [a, b] を 2 等分した 3 点を使用。
    次数 3 以下の多項式に対して厳密（超収束）。
    誤差: -(b-a)^5/90 * f^(4)(xi)
    """
    h = b - a
    m = (a + b) / 2
    return h / 6 * (f(a) + 4 * f(m) + f(b))


def simpson38(f, a: float, b: float) -> float:
    """Simpson 3/8 則: [a, b] を 3 等分した 4 点を使用。
    次数 3 以下の多項式に対して厳密。
    誤差: -(b-a)^5/80 * f^(4)(xi)
    """
    h = (b - a) / 3
    x1 = a + h
    x2 = a + 2 * h
    return (b - a) / 8 * (f(a) + 3 * f(x1) + 3 * f(x2) + f(b))


def boole(f, a: float, b: float) -> float:
    """Boole 則: [a, b] を 4 等分した 5 点を使用。
    次数 5 以下の多項式に対して厳密（超収束）。
    誤差: -2(b-a)^7/945 * f^(6)(xi)
    """
    h = (b - a) / 4
    x = [a + i * h for i in range(5)]
    w = [7, 32, 12, 32, 7]
    return (b - a) / 90 * sum(wi * f(xi) for wi, xi in zip(w, x))


# ============================================================
# 複合則（区間を N 等分して適用）
# ============================================================

def composite_trapezoid(f, a: float, b: float, N: int) -> float:
    """複合台形則: [a, b] を N 等分。収束次数 O(h^2), h=(b-a)/N。"""
    x = np.linspace(a, b, N + 1)
    y = f(x)
    h = (b - a) / N
    return h * (y[0] / 2 + np.sum(y[1:-1]) + y[-1] / 2)


def composite_simpson13(f, a: float, b: float, N: int) -> float:
    """複合 Simpson 1/3 則: N は偶数が必要。収束次数 O(h^4)。"""
    if N % 2 != 0:
        raise ValueError(f"N は偶数でなければなりません（N={N}）")
    x = np.linspace(a, b, N + 1)
    y = f(x)
    h = (b - a) / N
    # 重みパターン: 1, 4, 2, 4, 2, ..., 4, 1
    w = np.ones(N + 1)
    w[1:-1:2] = 4   # 奇数インデックス
    w[2:-2:2] = 2   # 偶数インデックス（両端以外）
    return h / 3 * np.dot(w, y)


def composite_simpson38(f, a: float, b: float, N: int) -> float:
    """複合 Simpson 3/8 則: N は 3 の倍数が必要。収束次数 O(h^4)。"""
    if N % 3 != 0:
        raise ValueError(f"N は 3 の倍数でなければなりません（N={N}）")
    x = np.linspace(a, b, N + 1)
    y = f(x)
    h = (b - a) / N
    # 重みパターン: 1, 3, 3, 2, 3, 3, 2, ..., 3, 3, 1
    w = np.ones(N + 1)
    for i in range(1, N):
        if i % 3 == 0:
            w[i] = 2
        else:
            w[i] = 3
    return 3 * h / 8 * np.dot(w, y)


def composite_boole(f, a: float, b: float, N: int) -> float:
    """複合 Boole 則: N は 4 の倍数が必要。収束次数 O(h^6)。"""
    if N % 4 != 0:
        raise ValueError(f"N は 4 の倍数でなければなりません（N={N}）")
    x = np.linspace(a, b, N + 1)
    y = f(x)
    h = (b - a) / N
    # 重みパターン（各パネル内）: 7, 32, 12, 32, 14, 32, 12, 32, 14, ..., 7
    w = np.zeros(N + 1)
    panel_w = [7, 32, 12, 32, 7]
    for k in range(N // 4):
        base = k * 4
        for j, wj in enumerate(panel_w):
            w[base + j] += wj
    return 2 * h / 45 * np.dot(w, y)


# ============================================================
# 図の作成
# ============================================================

def plot_method_visualization(f, a: float, b: float, N: int = 4, filename="fig_methods.png"):
    """各手法が関数をどう近似するかを可視化する（2×2 サブプロット）。"""
    x_fine = np.linspace(a, b, 500)
    y_fine = f(x_fine)

    methods = [
        ("台形則",       composite_trapezoid, "steelblue"),
        ("Simpson 1/3", composite_simpson13,  "darkorange"),
        ("Simpson 3/8", composite_simpson38,  "forestgreen"),
        ("Boole 則",    composite_boole,      "crimson"),
    ]
    # N が各手法の条件を満たすよう調整
    N_map = {"台形則": N, "Simpson 1/3": N, "Simpson 3/8": (N // 3) * 3 or 3, "Boole 則": (N // 4) * 4 or 4}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Newton-Cotes 各手法の可視化  [sin(x) on [0, π], N={N}]", fontsize=13)

    for ax, (name, func, color) in zip(axes.flat, methods):
        n = N_map[name]
        x_nodes = np.linspace(a, b, n + 1)
        y_nodes = f(x_nodes)

        ax.plot(x_fine, y_fine, "k-", lw=2, label="sin(x)")

        # 各パネルを塗りつぶして近似を可視化
        panel_size = {"台形則": 1, "Simpson 1/3": 2, "Simpson 3/8": 3, "Boole 則": 4}[name]
        for k in range(n // panel_size):
            xs = x_nodes[k * panel_size: k * panel_size + panel_size + 1]
            ys = y_nodes[k * panel_size: k * panel_size + panel_size + 1]
            x_fill = np.linspace(xs[0], xs[-1], 200)
            # ラグランジュ補間で多項式を構築して塗りつぶす
            poly_coeffs = np.polyfit(xs, ys, len(xs) - 1)
            y_poly = np.polyval(poly_coeffs, x_fill)
            ax.fill_between(x_fill, 0, y_poly, alpha=0.3, color=color)
            ax.plot(x_fill, y_poly, color=color, lw=1.2)

        ax.plot(x_nodes, y_nodes, "o", color=color, ms=5, label=f"ノード (N={n})")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend(fontsize=8)
        ax.set_xlim(a, b)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"  -> {filename} を保存しました")
    plt.close()


def plot_convergence(f, a: float, b: float, exact: float, N_values, filename="fig_convergence.png"):
    """各手法の収束を log-log プロットで比較する。"""
    funcs = {
        "台形則 O(h²)":       (composite_trapezoid, "steelblue",   "--"),
        "Simpson 1/3 O(h⁴)": (composite_simpson13,  "darkorange",  "-"),
        "Simpson 3/8 O(h⁴)": (composite_simpson38,  "forestgreen", "-."),
        "Boole 則 O(h⁶)":    (composite_boole,      "crimson",     ":"),
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (func, color, ls) in funcs.items():
        hs, errs = [], []
        for N in N_values:
            try:
                result = func(f, a, b, N)
            except ValueError:
                continue
            err = abs(result - exact)
            if err > 0:
                hs.append((b - a) / N)
                errs.append(err)
        if hs:
            ax.loglog(hs, errs, marker="o", color=color, ls=ls, lw=1.8, label=name)

    # 理論収束次数の参照線
    h_ref = np.array([hs[0], hs[-1]])  # 最後のメソッドの h 範囲を流用
    for order, label, alpha in [(2, "∝h²", 0.4), (4, "∝h⁴", 0.4), (6, "∝h⁶", 0.4)]:
        scale = errs[0] * (hs[0] ** (-order)) if hs else 1
        ax.loglog(h_ref, scale * h_ref ** order, "gray", lw=1, alpha=alpha, ls="-")
        ax.text(h_ref[-1] * 1.05, scale * h_ref[-1] ** order, label, color="gray", fontsize=9)

    ax.set_xlabel("h = (b−a)/N", fontsize=12)
    ax.set_ylabel("絶対誤差", fontsize=12)
    ax.set_title("Newton-Cotes 各手法の収束比較", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"  -> {filename} を保存しました")
    plt.close()


# ============================================================
# メイン: sin(x) を [0, π] で積分（厳密値 = 2）
# ============================================================

def _convergence_table(func_dict, f, a, b, exact, N_values):
    header = f"{'手法':<16} {'N':>5} {'結果':>18} {'誤差':>12} {'収束次数':>8}"
    print(header)
    print("-" * len(header))
    for name, func in func_dict.items():
        prev_err = None
        for N in N_values:
            try:
                result = func(f, a, b, N)
            except ValueError:
                continue
            err = abs(result - exact)
            rate = (np.log(prev_err / err) / np.log(2)) if prev_err and err > 0 else float("nan")
            prev_err = err
            print(f"{name:<16} {N:>5} {result:>18.12f} {err:>12.3e} {rate:>8.2f}")
        print()


if __name__ == "__main__":
    f = np.sin
    a, b = 0.0, np.pi
    exact = 2.0

    print("=" * 60)
    print("  Newton-Cotes 数値積分: sin(x) on [0, π]  (厳密値 = 2)")
    print("=" * 60)

    # --- 単区間ルールの比較 ---
    print("\n【単区間ルール（N=1）】")
    for name, func in [("台形則",        trapezoid),
                        ("Simpson 1/3",  simpson13),
                        ("Simpson 3/8",  simpson38),
                        ("Boole 則",     boole)]:
        result = func(f, a, b)
        print(f"  {name:<14}: {result:.10f}   誤差 = {abs(result - exact):.3e}")

    # --- 複合則の収束テーブル ---
    print("\n【複合則の収束テーブル】")
    funcs = {
        "台形則":       composite_trapezoid,
        "Simpson 1/3":  composite_simpson13,
        "Simpson 3/8":  composite_simpson38,
        "Boole 則":     composite_boole,
    }
    # LCM(1,2,3,4)=12 の倍数を使うと全手法で有効
    N_values = [12, 24, 48, 96, 192]
    _convergence_table(funcs, f, a, b, exact, N_values)

    # --- scipy との比較 ---
    try:
        from scipy.integrate import quad
        ref, _ = quad(f, a, b)
        print(f"scipy.integrate.quad 基準値: {ref:.15f}")
        result_boole = composite_boole(f, a, b, 32)
        print(f"複合 Boole (N=32):           {result_boole:.15f}  "
              f"差 = {abs(result_boole - ref):.3e}")
    except ImportError:
        print("（scipy がインストールされていないため比較をスキップ）")

    # --- 図の作成 ---
    print("\n【図の作成】")
    plot_method_visualization(f, a, b, N=4)
    plot_convergence(f, a, b, exact, N_values=[12, 24, 48, 96, 192, 384])
