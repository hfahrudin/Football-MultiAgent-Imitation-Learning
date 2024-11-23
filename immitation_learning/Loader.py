import numpy as np, pandas as pd, Utilities as util, os, json

class _Coordinated_Dataset:

    def __init__(self, ds_path, num_players=11):
        self.ds_path = ds_path
        self.all_ds_path = os.listdir(self.ds_path)
        self.num_players = num_players
        self.num_components = self.num_players

class DSSportsFormat(_Coordinated_Dataset):

    def __init__(self, ds_path, num_players=11):
        super().__init__(ds_path, num_players)

    def load_data(self, velocity=False):
        all_def = []
        all_off = []
        all_ball = []
        all_length = []
        for ds in self.all_ds_path:
            join_path = self.ds_path + ds
            df = pd.read_csv(join_path)
            if velocity:
                df = util.get_velocities(df)
            ball_df = df[["ball_x", "ball_y"]]
            off_ids = np.unique([c for c in df.columns if c[:3] in ('off', )])
            deff_ids = np.unique([c for c in df.columns if c[:3] in ('def', )])
            off_df = df[off_ids]
            deff_df = df[deff_ids]
            off = off_df.to_numpy()
            deff = deff_df.to_numpy()
            ball = ball_df.to_numpy()
            all_def.append(deff)
            all_off.append(off)
            all_ball.append(ball)
            lgth = deff_df.shape[0]
            all_length.append(lgth)

        return (all_off, all_def, all_ball, all_length)
