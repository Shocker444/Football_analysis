from inference import get_model
from dotenv import load_dotenv
import pickle
import os
import supervision as sv
from tqdm import tqdm

import numpy as np
Stride = 150

def extract_and_save(path, player_model, read=False, save_file = None):
    if read and os.path.exists(save_file):
        with open(save_file, 'rb') as f: # check if crops have already been extracted and saved
            crops = pickle.load(f)

        return crops
    
    videoinfo = sv.VideoInfo.from_video_path(path) # else extract and save crops
    frame_generator = sv.get_video_frames_generator(path, stride=Stride)

    player_id = 2
    crops = []

    for frame in tqdm(frame_generator, desc='collecting crops', total=videoinfo.total_frames/Stride):
        results = player_model.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(results)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == player_id]
        crops = crops + [
            sv.crop_image(frame, xyxy)
            for xyxy in detections.xyxy
        ]

    crops = [crop[0:crop.shape[0] // 2, :, :] for crop in crops]
    with open(save_file, 'wb') as f:
        pickle.dump(crops, f)


    return crops 


def resolve_goal_keeper_team(player_detec, goalkeeper_detec):
    goalkeeper_anchors = goalkeeper_detec.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    player_anchors = player_detec.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    avg_team0_pos = player_anchors[player_detec.class_id == 0].mean(axis=0)
    avg_team1_pos = player_anchors[player_detec.class_id == 1].mean(axis=0)

    goal_keeper_team_id = []

    for anchors in goalkeeper_anchors:
        dist0 = np.linalg.norm(anchors - avg_team0_pos)
        dist1 = np.linalg.norm(anchors - avg_team1_pos)

        goal_keeper_team_id.append(0 if dist0 < dist1 else 1)

    return np.array(goal_keeper_team_id, dtype=np.int32)


#c

#np.save(crops_path, crops)