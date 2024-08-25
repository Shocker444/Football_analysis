from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import pickle
import os
import cv2
from utils import get_center, get_width
from team_assigner import TeamColorAssigner

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames, batch=True, batch_size=20):
        if batch:
            detections = []
            for i in range(0, len(frames), batch_size):
                results = self.model.predict(frames[i:i+batch_size], conf=0.1)
                detections += results

            return detections
        
        detections = self.model.predict(frames, conf=0.1)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
        
        else:
            tracks = {
                'players': [],
                'referee': [],
                'ball': []
            }

            detections = self.detect_frames(frames)

    

            player_class = detections[0].boxes.cls == 2
            frame0_bbox = detections[0].boxes.xyxy[player_class]

            team_assigner = TeamColorAssigner()
            team_assigner.assign(frames[0], frame0_bbox)
            
    
            for frame_num, detect in enumerate(detections):

                class_names = detect.names
                class_inv = {v:k for k, v in class_names.items()}

                supervision_detec = sv.Detections.from_ultralytics(detect)
          
                for num, id in enumerate(supervision_detec.class_id):
                    if class_names[id] == 'goalkeeper':
                        supervision_detec.class_id[num] = class_inv['player']

                detec_with_tracks = self.tracker.update_with_detections(supervision_detec)

                
                tracks['players'].append({})
                tracks['referee'].append({})
                tracks['ball'].append({})

                for i, cls_id in enumerate(detec_with_tracks.class_id):
                    bbox = detec_with_tracks.xyxy[i]
                    track_id = detec_with_tracks.tracker_id[i]
                    if cls_id == class_inv['player']:
                        tracks['players'][frame_num][track_id] = {'bbox': bbox}
                        team = team_assigner.get_player_team(frames[frame_num], bbox, track_id)
                        tracks['players'][frame_num][track_id]['team'] = team
                        tracks['players'][frame_num][track_id]['team_color'] = team_assigner.team_colors[team[0]]
                    elif cls_id == class_inv['referee']:
                        tracks['referee'][frame_num][track_id] = {'bbox': bbox}

                    else:
                        tracks['ball'][frame_num][1] = {'bbox': bbox}

                for j, class_id in enumerate(supervision_detec.class_id):
                    bbox = supervision_detec.xyxy[j]
                    if class_id == class_inv['ball']:
                        tracks['ball'][frame_num][1] = {'bbox': bbox}    
               
            tracks['ball'] = self.interpolate_ball_positions(tracks['ball'])

            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)

        
        return tracks

    def draw_ellipse(self, frame, color, bbox, track_id, draw_rect=True):
        y2 = bbox[3]
        x_center, _ = get_center(bbox)

        x_width = get_width(bbox)

        cv2.ellipse(frame, center=(x_center, int(y2)), 
                    axes=(int(x_width), int(0.35 * x_width)),
                    angle=0.0,
                    startAngle=-45.0,
                    endAngle=235.0,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4)
        
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None and draw_rect:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame,
                        f'{track_id}',
                        (int(x1_text), int(y1_rect+15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2)

        
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center(bbox)

        triangle_points = np.array([[x, y], [x-10, y-20], [x+10, y-20]])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referee'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['team_color'], player['bbox'], track_id)

            for track_id, ref in referee_dict.items():
                frame = self.draw_ellipse(frame, (0, 255, 255), ref['bbox'], track_id, draw_rect=False)

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))
                
            
            output_frames.append(frame)

        return output_frames

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions