import numpy as np


class logiticsRgression():

        def __init__(self):
            pass



        def config(self):
            pass

        def sigmoid(self,z):
            return 1/(1+np.exp(z))



        def sign(self,x,shape):
            theta = np.ones(shape)
            return x*theta



        def