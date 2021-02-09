

class AlgoCEM:
    """
    The Algo class is an intermediate structure to unify various algorithms by collecting hyper-parameters
    """
    def __init__(self, study_name, policy, gamma, beta, n):
        self.study_name = study_name
        self.policy = policy
        self.gamma = gamma
        self.beta = beta
        self.n = n

    def prepare_batch(self, batch) -> None:
        """
        Applies reward transformations into the batch to prepare the computation of some gradient over these rewards
        :param batch: the batch on which we train
        :return: nothing
        """
        assert self.study_name in ['beta', 'sum', 'discount', 'normalize', 'baseline', 'nstep'], 'unsupported study name'
        if self.study_name == "beta":
            batch.exponentiate_rewards(self.beta)
        elif self.study_name == "sum":
            batch.sum_rewards()
