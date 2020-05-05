from .SRSolver import SRSolver
from .SRSolverWL import SRSolverWL

def create_solver(opt):
    if opt['mode'] == 'sr':
        solver = SRSolver(opt)

    elif opt['mode'] == 'sr_wl':
        solver = SRSolverWL(opt)
        
    else:
        raise NotImplementedError

    return solver