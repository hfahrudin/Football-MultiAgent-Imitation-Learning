
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

class RoleAssignment:

    def __init__(self, num_component=None, num_players=11, n_ind=2):
        self.num_players = num_players
        self.num_components = self.num_players if num_component is None else num_component
        self.n_ind = n_ind

    def assign_roles(self, moments, cmeans, event_lengths):
        all_moments = np.concatenate(moments, axis=0)
        all_moments_ = np.concatenate([all_moments[:, i:i + self.n_ind] for i in range(0, self.n_ind * self.num_players, self.n_ind)], axis=0)
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
            role_assignments_seq.append(role_assignments[start_id:start_id + i])
            start_id += i

        role_arranged = self.arrange_data(moments, role_assignments_seq, self.n_ind)
        return (
         role_assignments, role_arranged)

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
                repeats[:, range(i + 1, n_ind * self.num_players, n_ind)] += i + 1

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


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    """
        same as get_minibatches, except returns a generator
        inputs: An array of events, where each events is an array with sequence_length number of rows
        targets: target created by shifting 1 from inputs
        batchsize: desired batch size
        shuffle: Shuffle input data
    """
    if not len(inputs) == len(targets):
        raise AssertionError("inputs len: {0:} | targets len: {1:}".format(len(inputs), len(targets)))
    else:
        assert len(inputs) >= batchsize, "inputs len: {0:} | batch size: {1:}".format(len(inputs), batchsize)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
    for start_idx in range(0, len(inputs), batchsize):
        if shuffle:
            if len(inputs) % batchsize != 0 and start_idx + batchsize >= len(inputs):
                excerpt = indices[len(inputs) - batchsize:len(inputs)]
            else:
                excerpt = indices[start_idx:start_idx + batchsize]
        elif len(inputs) % batchsize != 0 and start_idx + batchsize >= len(inputs):
            excerpt = slice(len(inputs) - batchsize, len(inputs))
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield (
         np.array(inputs[excerpt]), np.array(targets[excerpt]))


def get_sequences(single_game, sequence_length, overlap, policy=None, n_fts=2, n_player=11):
    """ create events where each event is a list of sequences from
        single_game with required sequence_legnth and overlap

        single_game: A list of events
        sequence_length: the desired length of each event (a sequence of moments)
        overlap: how much overlap wanted for the sequence generation
        n_fts: individual player features e.g. n_fts = 2 => (x,y)
    """
    train = []
    target = []
    for i in single_game:
        i_len = len(i)
        if i_len < sequence_length:
            sequences = np.pad((np.array(i)), [(0, sequence_length - i_len), (0, 0)], mode="constant")
            if policy is None:
                targets = [
                 np.roll((sequences[:, :(n_player - 1) * n_fts + n_fts]), (-1), axis=0)[:-1, :]]
            else:
                targets = [
                 np.roll((sequences[:, policy * n_fts:policy * n_fts + n_fts]), (-1), axis=0)[:-1, :]]
            sequences = [
             sequences[:-1, :]]
        else:
            sequences = [np.array(i[-sequence_length:]) if j + sequence_length > i_len - 1 else np.array(i[j:j + sequence_length]) for j in range(0, i_len - overlap, sequence_length - overlap)]
            if policy is None:
                targets = [np.roll((k[:, :(n_player - 1) * n_fts + n_fts]), (-1), axis=0)[:-1, :] for k in sequences]
            else:
                targets = [np.roll((k[:, policy * n_fts:policy * n_fts + n_fts]), (-1), axis=0)[:-1, :] for k in sequences]
            sequences = [l[:-1, :] for l in sequences]
        train += sequences
        target += targets

    return (
     train, target)


def feature_roll(role_idx, x_curr, dup_ft=4, x_prev=None):
    k = dup_ft
    total_dup = k * 2
    curr_player_ft = x_curr[:, :, :44]
    curr_ball_ft = x_curr[:, :, -2:]
    if x_prev is not None:
        prev_player_ft = x_prev[:, :, :44]
        prev_ball_ft = x_prev[:, :, -2:]
    shape = list(x_curr.shape)
    shape[2] = 286
    new_feature = np.zeros(shape)
    goal_pos = [1.0, 0]
    active_player = curr_player_ft[:, :, role_idx * 2:role_idx * 2 + 2]
    ball_vel = np.zeros(curr_ball_ft.shape)
    if x_prev is not None:
        ball_vel = curr_ball_ft - prev_ball_ft
    new_ball_ft = np.concatenate((curr_ball_ft, ball_vel), axis=2)
    dist_def = np.zeros((shape[0], shape[1], 11))
    dist_off = np.zeros((shape[0], shape[1], 11))
    for i in range(22):
        new_feature[:, :, i * 13:i * 13 + 2] = curr_player_ft[:, :, i * 2:i * 2 + 2]
        goal_pos = [1.0, 0]
        if x_prev is None:
            new_feature[:, :, i * 13 + 2:i * 13 + 4] = 0
        else:
            new_feature[:, :, i * 13 + 2:i * 13 + 4] = curr_player_ft[:, :, i * 2:i * 2 + 2] - prev_player_ft[:, :, i * 2:i * 2 + 2]
        pos = new_feature[:, :, i * 13:i * 13 + 2]
        new_feature[:, :, i * 13 + 4] = ((pos[:, :, 0] - curr_ball_ft[:, :, 0]) ** 2 + (pos[:, :, 1] - curr_ball_ft[:, :, 1]) ** 2) ** 0.5
        a = pos[:, :, 0] - curr_ball_ft[:, :, 0]
        b = pos[:, :, 1] - curr_ball_ft[:, :, 1]
        c = new_feature[:, :, i * 13 + 4]
        new_feature[:, :, i * 13 + 5] = np.divide(a, c, out=(np.zeros_like(a)), where=(c != 0))
        new_feature[:, :, i * 13 + 6] = np.divide(b, c, out=(np.zeros_like(b)), where=(c != 0))
        new_feature[:, :, i * 13 + 7] = ((pos[:, :, 0] - goal_pos[0]) ** 2 + (pos[:, :, 1] - goal_pos[1]) ** 2) ** 0.5
        a = pos[:, :, 0] - goal_pos[0]
        b = pos[:, :, 1] - goal_pos[1]
        c = new_feature[:, :, i * 13 + 7]
        new_feature[:, :, i * 13 + 8] = np.divide(a, c, out=(np.zeros_like(a)), where=(c != 0))
        new_feature[:, :, i * 13 + 9] = np.divide(b, c, out=(np.zeros_like(b)), where=(c != 0))
        new_feature[:, :, i * 13 + 10] = ((pos[:, :, 0] - goal_pos[0]) ** 2 + (pos[:, :, 1] - goal_pos[1]) ** 2) ** 0.5
        a = pos[:, :, 0] - active_player[:, :, 0]
        b = pos[:, :, 0] - active_player[:, :, 1]
        c = new_feature[:, :, i * 13 + 10]
        new_feature[:, :, i * 13 + 11] = np.divide(a, c, out=(np.zeros_like(a)), where=(c != 0))
        new_feature[:, :, i * 13 + 12] = np.divide(b, c, out=(np.zeros_like(b)), where=(c != 0))
        if i < 11:
            dist_def[:, :, i] = new_feature[:, :, i * 13 + 10]
        else:
            dist_off[:, :, i - 11] = new_feature[:, :, i * 13 + 10]

    k_nearest_teammate = dist_def.argsort()[:, :, 1:k + 1]
    k_nearest_opponent = 11 + dist_off.argsort()[:, :, :k]
    k_combine = np.concatenate((k_nearest_teammate, k_nearest_opponent), axis=2)
    nearest_player_ft = np.zeros((shape[0], shape[1], k * 2 * 13))
    nearest_player_ft[:, :, :total_dup] = k_combine[:, :, :] * 13
    for z in range(1, 13):
        nearest_player_ft[:, :, z * total_dup:z * total_dup + total_dup] = nearest_player_ft[:, :, (z - 1) * total_dup:(z - 1) * total_dup + total_dup] + 1

    nearest_player_ft.sort()
    nearest_player_ft = nearest_player_ft.astype(int)
    final_feature = np.concatenate((new_feature, np.take(new_feature, nearest_player_ft), new_ball_ft), axis=2)
    return final_feature


def pred_sequence(agents, x):
    x_updated = np.copy(x)
    seq_length = x.shape[1]
    x_prev = None
    for s in range(seq_length - 1):
        for role_idx, agent in enumerate(agents):
            if agent is not None:
                x_curr = x_updated[:, s:s + 1, :]
                x_up = feature_roll(role_idx, x_curr, x_prev=x_prev)
                pred = agent(x_up)
                x_updated[:, s + 1, role_idx * 2:role_idx * 2 + 2] = pred
                x_prev = x_curr

    for role_idx, agent in enumerate(agents):
        agent.reset_states()

    return x_updated
