import torch


def fast_gradient_attack(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor, epsilon: float, norm: str = "2",
                         loss_fn=torch.nn.functional.cross_entropy):
    """
    Perform a single-step projected gradient attack on the input x.
    Parameters
    ----------
    logits: torch.Tensor of shape [B, K], where B is the batch size and K is the number of classes.
        The logits for each sample in the batch.
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the number of channels, and N is the image
       dimension.
       The input batch of images. Note that x.requires_grad must have been active before computing the logits
       (otherwise will throw ValueError).
    y: torch.Tensor of shape [B, 1]
        The labels of the input batch of images.
    epsilon: float
        The desired strength of the perturbation. That is, the perturbation (before clipping) will have a norm of
        exactly epsilon as measured by the desired norm (see argument: norm).
    norm: str, can be ["1", "2", "inf"]
        The norm with which to measure the perturbation. E.g., when norm="1", the perturbation (before clipping)
         will have a L_1 norm of exactly epsilon (see argument: epsilon).
    loss_fn: function
        The loss function used to construct the attack. By default, this is simply the cross entropy loss.

    Returns
    -------
    torch.Tensor of shape [B, C, N, N]: the perturbed input samples.

    """
    norm = str(norm)
    assert norm in ["1", "2", "inf"]

    ##########################################################
    # YOUR CODE HERE
    
    loss = loss_fn(logits, y)
    loss.backward()
    
    if norm == "2":
        batch_size = x.shape[0]
        grad_sign = x.grad.sign()
        
        grad_sign = grad_sign / torch.reshape(torch.norm(torch.reshape(grad_sign, shape=[batch_size, -1]), dim=1), shape=[batch_size, 1, 1, 1])
        
    
        delta = epsilon * grad_sign
        norm = torch.reshape(torch.norm(torch.reshape(delta, shape=[batch_size, -1]), dim=1), shape=[batch_size, 1, 1, 1])
        delta = torch.min(epsilon*delta/norm, delta)
        
        x_pert = x + delta
        x_pert = torch.clamp(x_pert, 0, 1)
        
    elif norm == "inf":      
        x_pert = x + epsilon * x.grad.sign()
        x_pert = torch.clamp(x_pert, 0, 1)
        
    ##########################################################

    return x_pert.detach()
