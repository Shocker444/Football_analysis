from utils import load_video, save_video
from trackers import Tracker
import supervision as sv


def main():
    video_frames = sv.get_video_frames_generator('08fd33_4.mp4')

    tracker = Tracker('training/weights/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='saved_tracks/tracks_stubs.pkl')
    
    output_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_frames, 'output_videos/output_vid.avi')

if __name__ == '__main__':
    main()
