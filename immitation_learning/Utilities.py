
import numpy as np, pandas as pd, pandas as pd, scipy.signal as signal

def get_file_name(path):
    name_ext = path.rsplit("/", 1)[-1]
    seperate_name = name_ext.rsplit(".", 1)
    return (seperate_name[0], seperate_name[1])


def unique_player_id(team):
    player_ids = np.unique([c[:-2] for c in team.columns if c[:3] in ('off', 'def')])
    return player_ids


def smoothen_feature(df, filter='sg', window=7, polyorder=1):
    for ft in df.keys():
        if filter == "sg":
            df[ft] = signal.savgol_filter((df[ft]), window_length=window, polyorder=polyorder)

    return df


def get_velocities(team, window=7, polyorder=1):
    player_ids = np.unique([c[:-1] for c in team.keys() if c[:1] in ('A', 'B')])
    for player in player_ids:
        vx = team[player + "x"]
        vy = team[player + "y"]
        v_x = signal.savgol_filter((vx.diff()), window_length=window, polyorder=polyorder)
        v_y = signal.savgol_filter((vy.diff()), window_length=window, polyorder=polyorder)
        team[player + "vx"] = v_x
        team[player + "vy"] = v_y

    team = team.replace(np.nan, 0)
    return team


def get_possession_naive(merge_df):
    poss_a = 0
    poss_b = 0
    row, col = merge_df.shape
    tmp = merge_df.iloc[:, 2:] - np.repeat((merge_df.iloc[:, :2].to_numpy()), ((col - 2) / 2), axis=1)
    possession = tmp.abs().idxmin(axis=1)
    for _, row in possession.iteritems():
        if isinstance(row, float):
            pass
        elif "A" in row:
            poss_a += 1
        else:
            poss_b += 1

    if poss_a > poss_b:
        return "A"
    else:
        return "B"


def additional_data(team, player_velocity=True, player_to_ball=True, player_to_goal=True):
    x_ball = team["0ball_0_x"]
    y_ball = team["0ball_0_y"]
    player_ids = np.unique([c[:-2] for c in team.columns if c[:2] in ('A_', 'B_')])
    dt = 0.2
    new_feature = pd.DataFrame()
    for player in player_ids:
        vx = team[player + "_x"]
        vy = team[player + "_y"]
        if player_to_goal:
            goal_dist_x = 1 - vx
            goal_dist_y = vy
            sqr = goal_dist_x ** 2 + goal_dist_y ** 2
            goal_dist = sqr ** 0.5
            new_feature[player + "_goal_dist"] = goal_dist
            new_feature[player + "_goal_cos"] = goal_dist_x / goal_dist
            new_feature[player + "_goal_sin"] = goal_dist_y / goal_dist
        if player_to_ball:
            ball_dist_x = x_ball - vx
            ball_dist_y = y_ball - vy
            sqr = ball_dist_x ** 2 + ball_dist_y ** 2
            ball_dist = sqr ** 0.5
            new_feature[player + "_ball_dist"] = ball_dist
            new_feature[player + "_ball_cos"] = ball_dist_x / ball_dist
            new_feature[player + "_ball_sin"] = ball_dist_y / ball_dist
        if player_velocity:
            v_x = vx.diff() / dt
            v_y = vy.diff() / dt
            new_feature[player + "_vx"] = v_x
            new_feature[player + "_vy"] = v_y
            new_feature[player + "_speed"] = np.sqrt(v_x ** 2 + v_y ** 2)

    new_feature = new_feature.replace(np.nan, 0)
    return new_feature


def flip_team(data):
    columns = [c for c in data.columns if c[-1].lower() in ('x', 'y')]
    data.loc[:, columns] *= -1
    return data


def change_possesion(data, teams_missing_players):
    data_keys = data.keys()
    new_keys = []
    off_key = []
    def_key = []
    for key in data_keys:
        if "off" in key:
            k = "def" + key[3:]
            new_keys.append(k)
            def_key.append(k)
        elif "def" in key:
            k = "off" + key[3:]
            new_keys.append(k)
            off_key.append(k)
        else:
            new_keys.append(key)

    tmp = teams_missing_players["off"]
    teams_missing_players["off"] = teams_missing_players["def"]
    teams_missing_players["def"] = tmp
    data = data.rename(columns=(dict(zip(data_keys, new_keys))))
    return data
