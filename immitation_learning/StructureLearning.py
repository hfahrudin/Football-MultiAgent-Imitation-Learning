import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from hmmlearn import hmm

class RoleAssignment:

    def __init__(self, n_iter, verbose, num_players=11):
        self.n_iter = n_iter
        self.verbose = verbose
        self.num_players = num_players
        self.num_components = self.num_players

    def train_hmm(self, moments, length, n_comp=None, n_ind=2, rand_seed=42):
        all_moments = np.concatenate(moments, axis=0)
        all_moments = np.concatenate([all_moments[(None[:None], i[:i + n_ind])] for i in range(0, n_ind * self.num_players, n_ind)], axis=0)
        lengths_repeat = np.concatenate([length for _ in range(self.num_players)], axis=0)
        if n_comp is None:
            n_comp = self.num_components
        model = hmm.GaussianHMM(n_components=n_comp, verbose=(self.verbose), n_iter=(self.n_iter), random_state=rand_seed)
        model.fit(all_moments, lengths_repeat)
        cmeans = model.means_
        covars = model.covars_
        state_sequence = model.predict(all_moments, lengths_repeat)
        state_sequence_ = state_sequence.reshape(self.num_players, -1).T
        return (state_sequence_, cmeans, covars, model)

    def assign_roles(self, all_moments_, all_moments, cmeans, event_lengths):
        ed = distance.cdist(all_moments_, cmeans, "euclidean")
        n = len(ed) // self.num_players
        assert n == all_moments.shape[0]

        def assign_index_(cm):
            row_ind, col_ind = linear_sum_assignment(cm)
            assignment = sorted((list(zip(row_ind, col_ind))), key=(lambda x: x[1]))
            return [j[0] for j in assignment]

        role_assignments = np.array([assign_index_(ed[np.arange(self.num_players) * n + i].T) for i in range(n)])
        role_assignments_seq = []
        start_id = 0
        for i in event_lengths:
            role_assignments_seq.append(role_assignments[start_id[:start_id + i]])
            start_id += i

        return (role_assignments, role_assignments_seq)

    def arrange_data(self, moments, role_assignments, n_ind=2):
        """
            moments: list of momnets e.g. [(38, 20), (15, 20), ..., ()]
            role_assignments: list of assignments
            components: number of components 
            n: numbner of players
            n_ind: features for individual player
        """

        def unstack_role(role):
            repeats = np.repeat((role * n_ind), ([n_ind] * self.num_players), axis=1).copy()
            for i in range(n_ind - 1):
                repeats[(None[:None], range(i + 1, n_ind * self.num_players, n_ind))] += i + 1

            return repeats

        droles = [unstack_role(i) for i in role_assignments]
        ro_single_game = []
        for i in range(len(moments)):
            ro_i = []
            for j in range(len(moments[i])):
                slots = np.zeros(n_ind * self.num_components)
                for k, v in enumerate(droles[i][j]):
                    slots[v] = moments[i][j][k]

                ro_i.append(slots)

            ro_single_game.append(np.array(ro_i))

        return ro_single_game
