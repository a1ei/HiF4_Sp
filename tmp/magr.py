import torch


def project_onto_l1_ball(x, eps=1.0):
    """
    Compute Euclidean projection onto the L1 ball for a batch.

      min ||x - u||_2 s.t. ||u||_1 <= eps

    Inspired by the corresponding numpy version by Adrien Gaidon.
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    row_ids = torch.arange(x.shape[0], device=x.device)
    theta = (cumsum[row_ids, rho - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)


def linfty_proximal(x, scale):
    """
    The proximal operator of l_infinity norm:

    Prox_{scale * |.|_infinity}(x) = x - scale * project_onto_l1_ball(x / scale)
    """
    assert scale != 0
    return x - scale * project_onto_l1_ball(x / scale)


def normalize_xtx(XtX):
    if torch.any(~torch.isfinite(XtX)):
        raise ValueError("MagR XtX contains non-finite values.")
    max_eigenvalue = torch.linalg.eigvalsh(XtX).amax()
    if not torch.isfinite(max_eigenvalue) or max_eigenvalue <= 0:
        raise ValueError("MagR XtX has invalid largest eigenvalue.")
    return XtX / max_eigenvalue


def W_proximal_preprocess_xtx(W, XtX, alpha=0.001, n_iter=200):
    W_hat = W.clone().T
    XtX = normalize_xtx(XtX)

    for _ in range(n_iter):
        W_hat = linfty_proximal(
            (W_hat - torch.matmul(XtX, W_hat - W.T)).T, alpha
        ).T

    return W_hat.T


def W_proximal_preprocess(W, X, device, alpha=0.001, n_iter=200):
    W_hat = W.clone().T

    m, n = X.shape

    U, s, Vt = torch.linalg.svd(X, full_matrices=False)
    del X
    s /= torch.max(s)
    S = torch.diag(s)
    if m > n:
        U = U[:, :n]
    elif m < n:
        Vt = Vt[:m, :]

    X = torch.mm(torch.mm(U, S), Vt)
    XtX = torch.matmul(X.T, X).to(device)

    for _ in range(n_iter):
        W_hat = linfty_proximal(
            (W_hat - torch.matmul(XtX, W_hat - W.T)).T, alpha
        ).T

    del XtX
    return W_hat.T


def project_onto_l1_ball_groupwise(x, eps=1.0):
    """
    Compute Euclidean projection onto the L1 ball groupwise.
    """
    batch_size, num_groups, group_size = x.shape
    x = x.view(batch_size * num_groups, group_size)

    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, group_size + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    row_ids = torch.arange(batch_size * num_groups, device=x.device)
    theta = (cumsum[row_ids, rho - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)

    return x.view(batch_size, num_groups, group_size)


def linfty_proximal_groupwise(x, scale, group_size=128):
    """
    The proximal operator of L-infinity norm applied groupwise.
    """
    assert scale != 0

    num_features = x.shape[1]

    if num_features % group_size != 0:
        raise ValueError("The number of features must be divisible by the group size.")

    num_groups = num_features // group_size

    x = x.view(-1, num_groups, group_size)
    proximal_result = x - scale * project_onto_l1_ball_groupwise(x / scale)

    return proximal_result.view(-1, num_features)


def W_proximal_preprocess_groupwise(
    W, X, device, alpha=0.0001, n_iter=200, group_size=128
):
    W_hat = W.clone().T

    m, n = X.shape

    U, s, Vt = torch.linalg.svd(X, full_matrices=False)
    del X
    s /= torch.max(s)
    S = torch.diag(s)
    if m > n:
        U = U[:, :n]
    elif m < n:
        Vt = Vt[:m, :]

    X = torch.mm(torch.mm(U, S), Vt)
    XtX = torch.matmul(X.T, X).to(device)

    for _ in range(n_iter):
        W_hat = linfty_proximal_groupwise(
            (W_hat - torch.matmul(XtX, W_hat - W.T)).T,
            scale=alpha,
            group_size=group_size,
        ).T

    del XtX
    return W_hat.T


def W_proximal_preprocess_groupwise_xtx(
    W, XtX, alpha=0.0001, n_iter=200, group_size=128
):
    W_hat = W.clone().T
    XtX = normalize_xtx(XtX)

    for _ in range(n_iter):
        W_hat = linfty_proximal_groupwise(
            (W_hat - torch.matmul(XtX, W_hat - W.T)).T,
            scale=alpha,
            group_size=group_size,
        ).T

    return W_hat.T
