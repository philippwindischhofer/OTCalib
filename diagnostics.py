import torch

def grad(obj, var):
    return torch.autograd.grad(obj, var, grad_outputs = torch.ones_like(obj), create_graph = True)[0]

def apply_transport(g_func, source):

    g_vals = g_func(source)
    transported = grad(g_vals, source)

    return transported

def normsq(obj, dim = 1):
    return 0.5 * torch.sum(torch.square(obj), dim = dim)

def get_duality_gap(f_func, g_func, source_data, target_data):

    # compute upper bound
    transport_vector = apply_transport(g_func, source_data) - source_data
    transport_cost = normsq(transport_vector)
    upper = torch.mean(transport_cost)

    # compute lower bound
    c_source_target = torch.mean(normsq(source_data)) + torch.mean(normsq(target_data))
    
    vals_g = g_func(source_data)
    grad_g = grad(vals_g, source_data)

    lower = c_source_target - \
            torch.mean(f_func(target_data)) - \
            torch.mean(torch.sum(grad_g * source_data, keepdim = True, dim = 1)) + \
            torch.mean(f_func(grad_g))

    # duality gap
    return upper - lower
