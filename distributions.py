import numpy as np


def double_geometric(p, size):
    ''' This is a distribution (1-p)^{|z|} p / (2-p)


        zero - p / (2-p)
        +ve : (1-p) / (2-p)
        -ve : (1-p) / (2-p)
        then its geometric (unshifted) for both +ve and -ve with probability p
    '''
    A = np.random.geometric(p, size=size) #+ve
    B = -np.random.geometric(p, size=size) #-ve
    Ax = np.random.binomial(1, 0.5, size=size)
    Bx = 1 - Ax # choosing A or B equally

    Zero = np.random.binomial(1, p / (2-p), size=size) # places where it will be 0

    final = (A*Ax + B*Bx)*(1-Zero)
    return final

    
