import random
from deap import base
from deap import creator
from deap import tools
from typing import Callable

def ensure_deap_types(maximize = True):
    """Create classes for Individuals and Fitness.

    Args:
        maximize (bool, optional): Whether individual instances created should have fitnesses 
                                   to be maximized or minimized. Defaults to True.

    Returns:
        fit_type : Class for fitness
        ind_type : Class for individual
    """
    fit_name = "FitnessMax" if maximize else "FitnessMin"
    ind_name = "IndividualMax" if maximize else "IndividualMin"
    
    if not hasattr(creator, fit_name):
        creator.create(fit_name, base.Fitness, weights = (1.0,) if maximize else (-1.0,))
        
    if not hasattr(creator, ind_name):
        creator.create(ind_name, list, fitness = getattr(creator, fit_name))
    fit_type = getattr(creator, fit_name)
    ind_type = getattr(creator, ind_name)
    
    return fit_type, ind_type    


def register_pop_factories(
    t : base.Toolbox,
    ind_type : type,
    init_func : Callable[..., float],
    t_attr_name : str,
    t_ind_name : str,
    t_pop_name : str,
    k : int = 10,
    ) -> None:
    """
    Creates DEAP toolbox function to generate an allele, an individual, and a population. The distribution 
    to sample the value of each allele from can be specified. Alleles are assumed to be floats. 
    Args:
        t : DEAP toolbox,
        ind_type : The individual class/type, 
        init_func : The function to sample alleles from
        t_attr_name : The name for the sampling function to be set in toolbox
        t_ind_name : The name for the individual generating function to be set in the toolbox
        t_pop_name : The name for the population generating function to be set in the toolbox
        k : The number of alleles per individual
    """

    if not hasattr(t, t_attr_name):
        t.register(t_attr_name, init_func)
    if not hasattr(t, t_ind_name):
        t.register(t_ind_name, tools.initRepeat, ind_type, getattr(t, t_attr_name), n=k)
    if not hasattr(t, t_pop_name):
        t.register(t_pop_name, tools.initRepeat, list, getattr(t, t_ind_name))
        
if __name__=="__main__":
    _, ind_type = ensure_deap_types(maximize= False)
    
    toolbox = base.Toolbox()
    register_pop_factories(toolbox, ind_type, random.random, "attr_float", "Individual", "Population", 10)
    pop = toolbox.Population(n = 100)
    

    
