
import cv2, numpy as np, json, pandas as pd, random, math, cmapy

class draw_cartesian_format:

    def __init__(self, field_dimen=(106.0, 68.0)):
        self.field_dim = field_dimen
        self.pallete = [[0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 0, 0]]

    def cart_to_px(self, cart_x, cart_y):
        new_cart_x = (cart_x + 1) / 2 * self.dim[1]
        new_cart_y = (-1 * cart_y + 1) / 2 * self.dim[0]
        return (int(new_cart_x), int(new_cart_y))

    def drawFrameNum(self, court_img, frame_num, title=None):
        (_, text_height), _ = cv2.getTextSize(str(frame_num), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        court_dim = court_img.shape
        cv2.putText(court_img, str(frame_num), (1, court_dim[0] - (1 + text_height)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,
                                                                                                                       255,
                                                                                                                       255), 2)
        if title is not None:
            cv2.putText(court_img, str(title), (1, 20 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,
                                                                                                       255,
                                                                                                       255), 2)
        return court_img

    def draw_one_frame(self, teams, court, thickness=10, radius=2):
        court_img = cv2.imread(court)
        self.dim = court_img.shape
        for i, team in enumerate(teams):
            color = self.pallete[i]
            for k, p in enumerate(team):
                x, y = self.cart_to_px(p[0], p[1])
                info = "Role" + str(k)
                cv2.circle(court_img, (x, y), radius, color, thickness)
                cv2.putText(court_img, info, (x, y), 0, 0.5, color)

            cv2.imshow("Frame", court_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            court_img = cv2.imread(court)

    def get_text_pos(self, x, y, info):
        text_size, _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        return (x - text_size[0] // 2, y + text_size[1] // 2)


class draw_structured_format(draw_cartesian_format):

    def __init__(self, field_dimen=(106.0, 68.0)):
        super().__init__(field_dimen)

    def drawPlayersOnCourt(self, court_img, structured_data, ghost_ft=None, radius=3):
        total_feature = structured_data.shape[0] - 2
        for i in range(0, total_feature, 2):
            if i < total_feature / 2:
                pallete = [
                 255, 0, 0]
                thickness = 10
                info = str(i // 2)
                if ghost_ft is not None:
                    ghost_pallete = [
                     165, 165, 165]
                    ghost_info = str(i // 2)
                    x, y = self.cart_to_px(ghost_ft[i], ghost_ft[i + 1])
                    text_pos = self.get_text_pos(x, y, ghost_info)
                    cv2.circle(court_img, (x, y), radius, ghost_pallete, thickness)
                    cv2.putText(court_img, ghost_info, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                                                                 255,
                                                                                                 255), 1)
            else:
                pallete = [
                 0, 0, 255]
                thickness = 10
                info = str(i // 2 - 11)
            x, y = self.cart_to_px(structured_data[i], structured_data[i + 1])
            text_pos = self.get_text_pos(x, y, info)
            cv2.circle(court_img, (x, y), radius, pallete, thickness)
            cv2.putText(court_img, info, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                                                   255,
                                                                                   255), 1)

        pallete = [
         255, 255, 255]
        thickness = 10
        x, y = self.cart_to_px(structured_data[44], structured_data[45])
        text_pos = self.get_text_pos(x, y, "b")
        cv2.circle(court_img, (x, y), radius, pallete, thickness)
        cv2.putText(court_img, "b", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (165,
                                                                              165,
                                                                              165), 1)
        return court_img

    def makeCourtVideo(self, structured_data, court_name, output_name, ghost_ft=None, title=None, fps=25, frame_drop_rate=1):
        totalFrames = structured_data.shape[0]
        court_img = cv2.imread(court_name)
        self.dim = court_img.shape
        writer = cv2.VideoWriter(output_name + "-court.webm", (cv2.VideoWriter_fourcc)(*"VP80"), fps / frame_drop_rate, (court_img.shape[1], court_img.shape[0]), True)
        for i in range(totalFrames):
            if i % frame_drop_rate == 0:
                court = court_img.copy()
                if ghost_ft is not None:
                    result = self.drawPlayersOnCourt(court, structured_data[i], ghost_ft[i])
                else:
                    result = self.drawPlayersOnCourt(court, structured_data[i])
                result = self.drawFrameNum(result, i, title)
                writer.write(result)

        writer.release()

class draw_json_format:

    def drawPlayersOnCourt2(self, court_img, team_info, teammarkcolor, radius=10):
        for ati in team_info:
            if ati[1] > 0:
                cv2.circle(court_img, (ati[0][0], ati[0][1]), radius, (teammarkcolor[ati[1] - 1]), thickness=2)
            else:
                cv2.circle(court_img, (ati[0][0], ati[0][1]), radius, [255, 255, 255], thickness=(-1))

        return court_img

    def drawFrameNum(self, court_img, frame_num):
        (_, text_height), _ = cv2.getTextSize(str(frame_num), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(court_img, str(frame_num), (1, 1 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                                                                     255,
                                                                                                     255), 1)
        return court_img

    def makeCourtVideo(self, json_data, original_video_name, court_name, output_name, src_pts, dst_pts, TeamMarkColors, frame_drop_rate=5):
        vs = cv2.VideoCapture(original_video_name)
        totalFrames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(vs.get(cv2.CAP_PROP_FPS))
        court_img = cv2.imread(court_name)
        with open(json_data) as f:
            all_data = json.load(f)
        player_mapper = all_data["Players"]
        if "FrameDropRate" in all_data:
            frame_drop_rate = all_data["FrameDropRate"]
        writer = cv2.VideoWriter(output_name + "-court.webm", (cv2.VideoWriter_fourcc)(*"VP80"), fps / frame_drop_rate, (court_img.shape[1], court_img.shape[0]), True)
        for i in range(totalFrames):
            if i % frame_drop_rate == 0:
                court = court_img.copy()
                result = self.drawPlayersOnCourt2(court, self.getResultAt(player_mapper, i), TeamMarkColors)
                result = self.drawFrameNum(result, i)
                writer.write(result)

        writer.release()

    def getResultAt(self, playerMapper, frame):
        result = []
        for player in playerMapper:
            pl = []
            if str(frame) in player["positions"]:
                pl.append(player["positions"][str(frame)])
                pl.append(player["team_number"])
                result.append(pl)

        return result
