from inference import get_model
from dotenv import load_dotenv
import torch
import os
import cv2
import supervision as sv
import numpy as np
from utils.utils import extract_and_save
from team_assigner.assigner import TeamAssigner
from annotations import ellipse_annotator, label_annotator, triangle_annotator, vertex_annotator
from generate_analysis import gen_analysis

# Suppress warnings and logging messages
import warnings
import logging

# Silence general warnings
warnings.filterwarnings("ignore")

# Silence inference telemetry/threading warnings
logging.getLogger("inference.usage_tracking.collector").setLevel(logging.ERROR)






load_dotenv()
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]" if torch.cuda.is_available() else "[CPUExecutionProvider]"

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam_path = input('Enter the football video path: ') 

    PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
    PLAYER_DETECTION_MODEL = get_model(
        model_id = PLAYER_DETECTION_MODEL_ID)
    PITCH_DETECTION_MODEL_ID = "football-field-detection-f07vi/14"
    PITCH_DETECTION_MODEL = get_model(PITCH_DETECTION_MODEL_ID)

    crops = extract_and_save(sam_path, PLAYER_DETECTION_MODEL, read=True, save_file='saved_crops/crops.pkl')


    team_assigner = TeamAssigner()
    team_assigner.fit(crops)

    gen_analysis(sam_path, 'analyzed2.mp4', 'pitch_analysis.mp4', PLAYER_DETECTION_MODEL, PITCH_DETECTION_MODEL, team_assigner, ellipse_annotator, triangle_annotator, 
                 label_annotator)

    
 

if __name__ == '__main__':
    main()
