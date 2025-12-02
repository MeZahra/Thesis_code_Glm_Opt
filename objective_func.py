def obj_func(X_beta, X_behave, C_bold, C_beta, C_task, W_pca):
    alpha_bold=1.0
    alpha_beta=1.0
    alpha_task=1.0
    n_comp = X_beta.shape[0]
    regularization = 1e-6
    solver_name = "MOSEK"

    # ecnter data
    X_beta = X_beta - np.mean(X_beta, axis=1, keepdims=True)
    valid_beh = np.isfinite(X_behave)
    mean_beh = np.nanmean(X_behave[valid_beh])
    X_behave = X_behave - mean_beh
    behave_norm = np.linalg.norm(np.nan_to_num(X_behave))

    C_total = alpha_task * np.diag(C_task) + alpha_bold * C_bold + alpha_beta * C_beta
    C_total = 0.5 * (C_total + C_total.T)
    C_total += regularization * np.eye(n_comp, dtype=np.float64)

    solver_const = getattr(cp, solver_name, None)
    w_var = cp.Variable(n_comp)

    constraints = []
    constraints.append(w_var >= 0)
    constraints.append(cp.sum(w_var) == 1)

    w_p =  w_var @ W_pca
    rho = 0.8
    C = rho **2 * behave_norm **2
    M = C * (X_beta @ X_beta.T) - X_beta @ (X_behave[:,None] @ X_behave[None, :]) @ X_beta.T
    constraints.append(w_p.T @ M @ w_p <= 0)

    objective = cp.Minimize(cp.quad_form(w_var, cp.psd_wrap(C_total)))
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=solver_const, warm_start=True)
    except cp.error.SolverError as exc:
        raise RuntimeError(f"Solver '{solver_name}' failed: {exc}") from exc

    if w_var.value is None:
        raise RuntimeError(f"Optimisation failed with status '{problem.status}'.")

    weights = np.array(w_var.value, dtype=np.float64).ravel()

    return weights