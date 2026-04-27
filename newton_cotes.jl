"""
Newton-Cotes 数値積分

閉じたニュートン・コーツ公式による数値積分の実装と可視化。
  - 台形則 (n=1)
  - Simpson 1/3 則 (n=2)
  - Simpson 3/8 則 (n=3)
  - Boole 則 (n=4)
"""

using LinearAlgebra
using Printf
using Plots

gr()  # PNG 保存に適した GR バックエンド

# ============================================================
# 単区間ルール
# ============================================================

trapezoid(f, a, b) = (b - a) / 2 * (f(a) + f(b))

simpson13(f, a, b) = (b - a) / 6 * (f(a) + 4f((a + b) / 2) + f(b))

function simpson38(f, a, b)
    h = (b - a) / 3
    (b - a) / 8 * (f(a) + 3f(a + h) + 3f(a + 2h) + f(b))
end

function boole(f, a, b)
    h = (b - a) / 4
    (b - a) / 90 * (7f(a) + 32f(a + h) + 12f(a + 2h) + 32f(a + 3h) + 7f(b))
end

# ============================================================
# 複合則（区間を N 等分して適用）
# ============================================================

function composite_trapezoid(f, a, b, N)
    xs = LinRange(a, b, N + 1)
    ys = f.(xs)
    h  = (b - a) / N
    h * (ys[1] / 2 + sum(ys[2:end-1]) + ys[end] / 2)
end

function composite_simpson13(f, a, b, N)
    N % 2 == 0 || throw(ArgumentError("N は偶数でなければなりません (N=$N)"))
    xs = LinRange(a, b, N + 1)
    ys = f.(xs)
    h  = (b - a) / N
    w  = fill(1.0, N + 1)
    w[2:2:end-1] .= 4
    w[3:2:end-2] .= 2
    h / 3 * dot(w, ys)
end

function composite_simpson38(f, a, b, N)
    N % 3 == 0 || throw(ArgumentError("N は 3 の倍数でなければなりません (N=$N)"))
    xs = LinRange(a, b, N + 1)
    ys = f.(xs)
    h  = (b - a) / N
    w  = fill(1.0, N + 1)
    for i in 2:N
        w[i] = (i - 1) % 3 == 0 ? 2.0 : 3.0
    end
    3h / 8 * dot(w, ys)
end

function composite_boole(f, a, b, N)
    N % 4 == 0 || throw(ArgumentError("N は 4 の倍数でなければなりません (N=$N)"))
    xs = LinRange(a, b, N + 1)
    ys = f.(xs)
    h  = (b - a) / N
    w  = zeros(N + 1)
    for k in 0:(N ÷ 4 - 1)
        for (j, wj) in enumerate([7, 32, 12, 32, 7])
            w[k * 4 + j] += wj
        end
    end
    2h / 45 * dot(w, ys)
end

# ============================================================
# 収束テーブル
# ============================================================

function print_convergence_table(f, a, b, exact, N_values)
    methods = [("Trapezoid",   composite_trapezoid),
               ("Simpson 1/3", composite_simpson13),
               ("Simpson 3/8", composite_simpson38),
               ("Boole",       composite_boole)]

    hdr = @sprintf("%-14s %5s %18s %12s %8s", "Method", "N", "Result", "Error", "Rate")
    println(hdr)
    println("-"^length(hdr))

    for (name, func) in methods
        prev_err = NaN
        for N in N_values
            result = try func(f, a, b, N) catch; continue end
            err    = abs(result - exact)
            if isnan(prev_err)
                @printf("%-14s %5d %18.12f %12.3e %8s\n", name, N, result, err, "  ---")
            else
                @printf("%-14s %5d %18.12f %12.3e %8.2f\n", name, N, result, err, log2(prev_err / err))
            end
            prev_err = err
        end
        println()
    end
end

# ============================================================
# ラグランジュ補間（可視化用）
# ============================================================

function lagrange_eval(xs, ys, t)
    n = length(xs)
    sum(ys[i] * prod((t - xs[j]) / (xs[i] - xs[j]) for j in 1:n if j ≠ i) for i in 1:n)
end

# ============================================================
# 図1: 各手法の近似の可視化（2×2 サブプロット）
# ============================================================

function plot_method_visualization(f, a, b; N=4, filename="fig_methods.png")
    x_fine = collect(LinRange(a, b, 500))
    y_fine = f.(x_fine)

    configs = [("Trapezoid Rule", 1, :steelblue),
               ("Simpson's 1/3",  2, :darkorange),
               ("Simpson's 3/8",  3, :forestgreen),
               ("Boole's Rule",   4, :crimson)]

    # 各手法の条件（N の倍数）に合わせて調整
    Ns_adj = [N, N, max(div(N, 3) * 3, 3), max(div(N, 4) * 4, 4)]

    subplots = map(zip(configs, Ns_adj)) do ((title, panel_size, color), n)
        x_nodes = collect(LinRange(a, b, n + 1))
        y_nodes = f.(x_nodes)

        p = plot(x_fine, y_fine; color=:black, lw=2, label="sin(x)",
                 title=title, xlabel="x", ylabel="f(x)",
                 xlims=(a, b), ylims=(-0.1, 1.25),
                 grid=true, gridalpha=0.3)

        for k in 0:(n ÷ panel_size - 1)
            idx      = (k * panel_size + 1):(k * panel_size + panel_size + 1)
            xs_panel = x_nodes[idx]
            ys_panel = y_nodes[idx]
            x_fill   = collect(LinRange(xs_panel[1], xs_panel[end], 100))
            y_poly   = lagrange_eval.(Ref(xs_panel), Ref(ys_panel), x_fill)
            plot!(p, x_fill, y_poly;
                  fill=0, fillalpha=0.3, fillcolor=color,
                  color=color, lw=1.2, label=false)
        end

        scatter!(p, x_nodes, y_nodes; color=color, ms=5, label="nodes (N=$n)")
        p
    end

    fig = plot(subplots..., layout=(2, 2), size=(900, 700))
    savefig(fig, filename)
    println("  -> $filename を保存しました")
end

# ============================================================
# 図2: 収束比較（log-log プロット）
# ============================================================

function plot_convergence(f, a, b, exact;
                          N_values=[12, 24, 48, 96, 192, 384],
                          filename="fig_convergence.png")
    methods = [("Trapezoid O(h²)",   composite_trapezoid, :steelblue,   :dash),
               ("Simpson 1/3 O(h⁴)", composite_simpson13, :darkorange,  :solid),
               ("Simpson 3/8 O(h⁴)", composite_simpson38, :forestgreen, :dashdot),
               ("Boole O(h⁶)",       composite_boole,     :crimson,     :dot)]

    p = plot(; xscale=:log10, yscale=:log10,
               xlabel="h = (b-a)/N", ylabel="Absolute Error",
               title="Newton-Cotes Convergence Comparison",
               legend=:bottomright, grid=true, gridalpha=0.3, size=(700, 550))

    all_hs = Float64[]; all_errs = Float64[]

    for (name, func, color, ls) in methods
        hs = Float64[]; errs = Float64[]
        for N in N_values
            result = try func(f, a, b, N) catch; continue end
            err = abs(result - exact)
            err > 0 && (push!(hs, (b - a) / N); push!(errs, err))
        end
        isempty(hs) && continue
        plot!(p, hs, errs; marker=:circle, color=color, linestyle=ls, lw=1.8, label=name)
        append!(all_hs, hs); append!(all_errs, errs)
    end

    # 参照線（O(h²), O(h⁴), O(h⁶)）
    if !isempty(all_hs)
        h_range = [minimum(all_hs), maximum(all_hs)]
        e_top   = maximum(all_errs)
        h_top   = maximum(all_hs)
        for order in [2, 4, 6]
            scale = e_top / h_top^order
            plot!(p, h_range, scale .* h_range .^ order;
                  color=:gray, lw=1, linestyle=:dash, alpha=0.5, label=false)
        end
    end

    savefig(p, filename)
    println("  -> $filename を保存しました")
end

# ============================================================
# メイン
# ============================================================

function main()
    f     = sin
    a, b  = 0.0, π
    exact = 2.0

    println("="^60)
    println("  Newton-Cotes 数値積分: sin(x) on [0, π]  (厳密値 = 2)")
    println("="^60)

    println("\n【単区間ルール（N=1）】")
    for (name, func) in [("台形則",       trapezoid),
                          ("Simpson 1/3", simpson13),
                          ("Simpson 3/8", simpson38),
                          ("Boole 則",    boole)]
        result = func(f, a, b)
        @printf("  %-14s: %.10f   誤差 = %.3e\n", name, result, abs(result - exact))
    end

    println("\n【複合則の収束テーブル】")
    print_convergence_table(f, a, b, exact, [12, 24, 48, 96, 192])

    println("【図の作成】")
    plot_method_visualization(f, a, b; N=4)
    plot_convergence(f, a, b, exact)
end

main()
