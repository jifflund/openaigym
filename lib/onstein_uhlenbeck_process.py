import numpy as np

class OrnsteinUhlenbeckProcess(object):

    @staticmethod
    def function(x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)

