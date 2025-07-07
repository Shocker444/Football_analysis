import numpy as np
import cv2

class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(self.source, self.target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        try:
            points = points.reshape(-1, 1, 2).astype(np.float32)
            points = cv2.perspectiveTransform(points, self.m)
            points = points.reshape(-1, 2).astype(np.float32)
        
        except Exception as e:
            points = np.array([[0, 0]], dtype=np.float32)
        
        return points