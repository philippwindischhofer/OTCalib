import torch
import numpy as np

# see details at https://arxiv.org/abs/1609.07152
class ICNN(torch.nn.Module):

    def __init__(self
        , number_nonconvex_inputs
        , number_convex_inputs
        , nonconvex_activation
        , convex_activation
        , nonconvex_layersizes
        , convex_layersizes
        ):

        super(ICNN, self).__init__()

        assert len(nonconvex_layersizes) + 1 == len(convex_layersizes)

        self.nlayers = len(convex_layersizes) - 1

        self.g = [convex_activation for i in range(self.nlayers)]
        self.gtilde = [nonconvex_activation for i in range(self.nlayers - 1)]


        # simple matrix mul
        def W(x, y):
            return torch.nn.Linear(x, y, bias=False)

        # and full linear layer with bias
        def L(x, y):
            return torch.nn.Linear(x, y)


        # more-or-less following the nomenclature from
        # arXiv:1609.07152

        # shorthand:
        # zsize = convex layer sizes
        # usize = nonconvex layer sizes
        zsize = convex_layersizes
        usize = nonconvex_layersizes
        ysize = zsize[0]


        Wzz = []
        Wyz = []
        Luz = []
        Luz1 = []
        Luy = []
        Luutilde = []


        for lay in range(self.nlayers):
            Wzz.append(W(zsize[lay], zsize[lay+1]))
            Wyz.append(W(ysize, zsize[lay+1]))
            Luz.append(L(usize[lay], zsize[lay]))
            Luz1.append(L(usize[lay], zsize[lay+1]))
            Luy.append(L(usize[lay], ysize))

        for lay in range(self.nlayers - 1):
            Luutilde.append(L(usize[lay], usize[lay+1]))


        self.Wzz = torch.nn.ModuleList(Wzz)
        self.Wyz = torch.nn.ModuleList(Wyz)
        self.Luz = torch.nn.ModuleList(Luz)
        self.Luz1 = torch.nn.ModuleList(Luz1)
        self.Luy = torch.nn.ModuleList(Luy)
        self.Luutilde = torch.nn.ModuleList(Luutilde)

        # the authors set the weights in the first layer to zero.
        for p in Wzz[0].parameters():
            p.data.copy_(torch.zeros_like(p.data))
            p.requires_grad = False


    def forward(self, xs, ys):
        ui = xs
        zi = torch.zeros_like(ys)

        for i in range(self.nlayers):
            zi = \
              self.g[i](
                  self.Wzz[i](zi * torch.relu(self.Luz[i](ui))) \
                + self.Wyz[i](ys * self.Luy[i](ui)) \
                + self.Luz1[i](ui)
              )

            # no need to update ui the last time through.
            if i < self.nlayers - 1:
                ui = self.gtilde[i](self.Luutilde[i](ui))

        return zi


    def enforce_convexity(self):

        # apply param = max(0, param) = relu(param) to all parameters that need to be nonnegative
        for W in self.Wzz:
            for w in W.parameters():
                w.data.copy_(torch.relu(w.data))


    def get_convexity_regularisation_term(self):

        L2_reg = 0.0

        for W in self.Wzz:
            for w in W.parameters():
                L2_reg += torch.sum(torch.square(torch.relu(-w.data)))

        return L2_reg


# need a smooth version to make sure the transport potential has a smooth gradient
# NEED TO FIND SOMETHING MORE EFFICIENT HERE
def smooth_leaky_ReLU(x, a):
    sqrtpi = np.sqrt(np.pi)
    return 0.5 * ((1 - a) * torch.exp(-torch.square(x)) + sqrtpi * x * (1 + torch.erf(x) + a * torch.erfc(x)))

def construct_smooth_circle_arc(a):

    # require: 0 <= a < 1

    sq2 = 2 * np.sqrt(2)

    # compute constants
    x0 = (1 + 3 * a**2 + sq2 * a * np.sqrt(1 + a**2)) / ((a - 1) * (a + 1))
    R = (sq2 * np.sqrt(1 + a**2) * np.sqrt(1 + 3 * a**2 + sq2 * a * np.sqrt(1 + a**2))) / ((1 - a) * (1 + a))

    C0 = np.sqrt(R**2 - x0**2) - np.sqrt(R**2 - (1 + x0)**2)
    C1 = np.sqrt(R**2 - x0**2)
    C2 = np.sqrt(R**2 - x0**2) - np.sqrt(R**2 - (1 - x0)**2)

    def act(x):

        f0 = a * (x + 1) + C0 # x < -1
        f1 = C1 - torch.sqrt(R**2 - torch.square(x - x0)) # -1 < x < 1
        f2 = x - 1 + C2 # x > 1

        return torch.where(x > 1, f2, torch.where(x < -1, f0, f1))

    return act
