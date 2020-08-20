import torch
import numpy as np


class HawkesModel:

    def __init__(self, excitation, verbose=False, device='cpu'):
        """
        Initialize the model

        Arguments:
        ----------
        prior : Prior
            Prior object
        excitation: excitation
            Excitation object
        """
        self.excitation = excitation
        self.n_jumps = None
        self.dim = None
        self.n_params = None
        self.n_var_params = None
        self._fitted = False
        self.verbose = verbose
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.events_samples = None
        self.end_time = {}
        self._cache = {}
        self._cache_integral = {}

    #def __setitem__(self, k, v):
    #    self.k = v

    #def set_data(self, events, end_time):
    def set_data(self, events):
        """
        Set the data for the model
        改造数据结构，一次接收多个样本。
        """
        '''
        assert isinstance(events[0], torch.Tensor)
        # Set various util attributes
        self.dim = len(events)
        self.n_params = self.dim * (self.dim + 1)
        self.n_var_params = 2 * self.n_params
        self.n_jumps = sum(map(len, events))
        self.end_time = max([max(num) for num in events if len(num) > 0])
        self.events = events
        if not self._fitted:
            self._init_cache()
        self._fitted = True
        '''
        assert isinstance(events[0][0], torch.Tensor)
        self.events_samples = len(events)
        self.dim = len(events[0])
        self.n_params = self.dim * (self.dim + 1)
        #n_var_params = 2 * len(events[0]) * (len(events[0]) + 1)
        self.n_var_params = 2*self.n_params
        self.n_jumps = len(events)*sum(map(len, events[0]))     #每个样本的事件数一样，所以用了样本数乘以每个样本的事件数。
        for i in range(self.events_samples):
            #zc_debut = max([max(num) for num in events[i] if len(num) > 0])
            #print("i is :",i)
            self.end_time[i] = max([max(num) for num in events[i] if len(num) > 0])
        self.events = events
        if not self._fitted:
            self._init_cache()
        self._fitted = True


    def _init_cache(self):
        """
        caching the required computations

        cache[i][j,0,k]: float
            sum_{t^j < t^i_k} phi(t^i_k - t^j)
            This is used in k^th timestamp of node i, i.e., lambda_i(t^i_k)
        cache_integral: float
            used in the integral of intensity
        """
        '''
        self._cache = [torch.zeros(
            (self.dim, self.excitation.M, len(events_i)), dtype=torch.float64, device=self.device)
            for events_i in self.events]
        for i in range(self.dim):
            for j in range(self.dim):
                if self.verbose:
                    print(f"\rInitialize cache {i*self.dim+j+1}/{self.dim**2}     ", end='')
                id_end = np.searchsorted(
                    self.events[j].cpu().numpy(),
                    self.events[i].cpu().numpy())
                id_start = np.searchsorted(
                    self.events[j].cpu().numpy(),
                    self.events[i].cpu().numpy() - self.excitation.cut_off)
                for k, time_i in enumerate(self.events[i]):
                    t_ij = time_i - self.events[j][id_start[k]:id_end[k]]
                    kappas = self.excitation.call(t_ij).sum(-1)  # (M)
                    self._cache[i][j, :, k] = kappas
        if self.verbose:
            print()

        self._cache_integral = torch.zeros((self.dim, self.excitation.M),
                                           dtype=torch.float64, device=self.device)
        for j in range(self.dim):
            t_diff = self.end_time - self.events[j]
            integ_excit = self.excitation.callIntegral(t_diff).sum(-1)  # (M)
            self._cache_integral[j, :] = integ_excit
        '''
        '''
        改造成一次计算多个样本
        '''
        for s in range(self.events_samples):
            self._cache[s] = [torch.zeros(
                (self.dim, self.excitation.M, len(events_i)), dtype=torch.float64, device=self.device)
                for events_i in self.events[s]]
            #print(self._cache)
            for i in range(self.dim):
                for j in range(self.dim):
                    if self.verbose:
                        print(f"\rInitialize cache {i * self.dim + j + 1}/{self.dim ** 2}     ", end='')
                    id_end = np.searchsorted(
                        self.events[s][j].cpu().numpy(),
                        self.events[s][i].cpu().numpy())
                    id_start = np.searchsorted(
                        self.events[s][j].cpu().numpy(),
                        self.events[s][i].cpu().numpy() - self.excitation.cut_off)
                    for k, time_i in enumerate(self.events[s][i]):
                        t_ij = time_i - self.events[s][j][id_start[k]:id_end[k]]
                        kappas = self.excitation.call(t_ij).sum(-1)  # (M)
                        self._cache[s][i][j, :, k] = kappas
            if self.verbose:
                print()

            self._cache_integral[s] = torch.zeros((self.dim, self.excitation.M),
                                               dtype=torch.float64, device=self.device)
            for j in range(self.dim):
                t_diff = self.end_time[s] - self.events[s][j]
                integ_excit = self.excitation.callIntegral(t_diff).sum(-1)  # (M)
                self._cache_integral[s][j, :] = integ_excit


    def log_likelihood(self, mu, W, epsilon_noise = None):
        """
        Log likelihood of Hawkes Process for the given parameters mu and W

        Arguments:
        ----------
        mu : torch.Tensor
            (dim x 1)
            Base intensities
        W : torch.Tensor
            (dim x dim x M) --> M is for the number of different excitation functions
            The weight matrix.
        """
        #log_like = [[0.0]*self.events_samples]
        log_like = torch.zeros(self.events_samples)
        #print('small W:',W)
        '''
        for i in range(self.dim):
            # W[i] (dim x M)
            # _cache[i] (dim x M X len(events[i]))
            intens = torch.log(mu[i] + (W[i].unsqueeze(2) * self._cache[i]).sum(0).sum(0))
            log_like += intens.sum()
        log_like -= self._integral_intensity(mu, W)
        return log_like
        '''
        #intens = [[0.0]*self.dim for i in range(self.events_samples)]
        #print (intens)
        min_intens = [0.0]*self.events_samples
        if epsilon_noise == None:
            #先固定epsilon_noise，给一个初始值（mean=-5.0,sigma=0.1不中，会出现Nan的现象，所以调整到mean=-10,sigma为0.01）。
            #epsilon_noise = torch.tensor(np.random.lognormal(mean=-1.0, sigma=0.1, size=self.events_samples))
            epsilon_noise = np.random.lognormal(mean=-3.0, sigma=0.1, size=self.events_samples)

        intens_sum = []
        integral_instesity = []
        intens_oral_sum = []

        for s in range(self.events_samples):

            intens_oral = [0.0] * self.dim
            intens = [0.0] * self.dim
            intens_every = [0.0] * self.dim
            intens_every_oral = [0.0] * self.dim

            for i in range(self.dim):
                intens_every[i] = mu[i] + (W[i].unsqueeze(2) * self._cache[s][i]).sum(0).sum(0) - epsilon_noise[s]
                intens_every_oral[i] = mu[i] + (W[i].unsqueeze(2) * self._cache[s][i]).sum(0).sum(0)
                if intens_every[i].sum() <= 0:
                    print('/ns is:', s)
                    print('/ni is:', i)
                    print('/nepsilon is:',epsilon_noise[i])
                intens[i] = torch.log(intens_every[i])
                intens_oral[i] = intens_every_oral[i].sum()
                log_like[s] += intens[i].sum()

            log_like[s] -= self._integral_intensity(mu, W, s)
            integral_instesity.append(self._integral_intensity(mu, W, s))
            intens_oral_sum.append(intens_oral)

            #print(intens)
            #min_intens[s] = min(intens_sum)
        log_like_sum = log_like.sum(0)

        return log_like_sum,intens_oral_sum,integral_instesity,self.end_time #返回log_likelihood，最小强度函数值min_intens,log_like的前半段和最大时间end_time


    def _integral_intensity(self, mu, W, s):
        """
        Integral of intensity function

        Argument:
        ---------
        node_i: int
            Node id
        """
        integ_ints = (W * self._cache_integral[s].unsqueeze(0)).sum(1).sum(1)
        integ_ints += self.end_time[s] * mu
        return integ_ints.sum()
