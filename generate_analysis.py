import supervision as sv
from tqdm import tqdm
import numpy as np
from transform_perspective import ViewTransformer
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from utils.utils import resolve_goal_keeper_team
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch
import matplotlib.pyplot as plt

CONFIG = SoccerPitchConfiguration()



def gen_analysis(vid_path, target_vid_path, target_vid_path2, player_model, pitch_detection_model, team_assigner,
                 ellipse_annotator, triangle_annotator, label_annotator):

    videoinfo = sv.VideoInfo.from_video_path(vid_path)
    vidsink = sv.VideoSink(target_vid_path, video_info=videoinfo)

    pitch_width = 1300
    pitch_height = 800

    pitch_vid_info = sv.VideoInfo(
                fps=videoinfo.fps,         # keep same FPS as original video
                width=pitch_width,
                height=pitch_height
        )
    
    vidsink2 = sv.VideoSink(target_vid_path2, video_info=pitch_vid_info, codec='mp4v')
    tracker = sv.ByteTrack()
    tracker.reset()

    framegen = sv.get_video_frames_generator(vid_path)

    with vidsink, vidsink2:
        for frame in tqdm(framegen, desc='Generating analysis', total = videoinfo.total_frames):
            results = player_model.infer(frame, confidence=0.3)[0]
            detections = sv.Detections.from_inference(results)

            player_id = 2
            ref_id = 3
            goal_id = 1
            ball_id = 0

            ball_detections = detections[detections.class_id == ball_id]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            other_detections = detections[detections.class_id != ball_id]
            other_detections = other_detections.with_nms(threshold=0.5, class_agnostic=True)

            other_detections = tracker.update_with_detections(other_detections)

            player_detections = other_detections[other_detections.class_id == player_id]
            player_crops = [sv.crop_image(frame, xyxy)
                                for xyxy in player_detections.xyxy]
            player_detections.class_id = team_assigner.predict(player_crops)

            goalkeeper_detections = other_detections[other_detections.class_id == goal_id]
            goalkeeper_detections.class_id = resolve_goal_keeper_team(player_detections,
                                                                        goalkeeper_detections)

            ref_detections = other_detections[other_detections.class_id == ref_id]
            ref_detections.class_id -= 1

            other_detections = sv.Detections.merge([player_detections, goalkeeper_detections, ref_detections])

            labels = [f"{track_id}" for track_id in other_detections.tracker_id]

            annotated_frame = frame.copy()
            annotated_frame = ellipse_annotator.annotate(annotated_frame, other_detections)
            annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections)
            annotated_frame = label_annotator.annotate(annotated_frame, other_detections, labels=labels)
            vidsink.write_frame(annotated_frame)



            ## Generating pitch analysis
            result = pitch_detection_model.infer(frame, confidence=0.3)[0]
            key_points = sv.KeyPoints.from_inference(result)


            filter = key_points.confidence[0] > 0.5
            frame_reference_points = key_points.xy[0][filter]
            frame_reference_key_points = sv.KeyPoints(xy=frame_reference_points[np.newaxis, ...])
            pitch_reference_points = np.array(CONFIG.vertices)[filter]

            view_transformer = ViewTransformer(source=frame_reference_points,
                                            target=pitch_reference_points)
            

            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = view_transformer.transform_points(frame_ball_xy)

            frame_players_xy = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_players_xy = view_transformer.transform_points(frame_players_xy)

            frame_refs_xy = ref_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_refs_xy = view_transformer.transform_points(frame_refs_xy)

            frame_goal_xy = goalkeeper_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_goal_xy = view_transformer.transform_points(frame_goal_xy)


            pitch = draw_pitch(config=CONFIG)

            pitch = draw_points_on_pitch(config=CONFIG,
                                        xy = pitch_ball_xy,
                                        face_color = sv.Color.WHITE,
                                        edge_color = sv.Color.BLACK,
                                        radius=10,
                                        pitch=pitch)
            
            pitch = draw_points_on_pitch(config=CONFIG,
                                        xy=pitch_players_xy[player_detections.class_id == 0],
                                        face_color=sv.Color.from_hex('#00ff1b'),
                                        edge_color=sv.Color.BLACK,
                                        radius=10,
                                        pitch=pitch)
            
            pitch = draw_points_on_pitch(config=CONFIG,
                                        xy=pitch_players_xy[player_detections.class_id == 1],
                                        face_color=sv.Color.from_hex('#0ed0ff'),
                                        edge_color=sv.Color.BLACK,
                                        radius=10,
                                        pitch=pitch)
            
            pitch = draw_points_on_pitch(config=CONFIG,
                                        xy=pitch_goal_xy[goalkeeper_detections.class_id == 0],
                                        face_color=sv.Color.from_hex('#00ff1b'),
                                        edge_color=sv.Color.BLACK,
                                        radius=10,
                                        pitch=pitch)
            
            pitch = draw_points_on_pitch(config=CONFIG,
                                        xy=pitch_goal_xy[goalkeeper_detections.class_id == 1],
                                        face_color=sv.Color.from_hex('#00ff1b'),
                                        edge_color=sv.Color.BLACK,
                                        radius=10,
                                        pitch=pitch)
            
            pitch = draw_points_on_pitch(config=CONFIG,
                                        xy=pitch_refs_xy,
                                        face_color=sv.Color.from_hex('#ffae00'),
                                        edge_color=sv.Color.BLACK,
                                        radius=10,
                                        pitch=pitch)
            
            
            vidsink2.write_frame(pitch)



