def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_width(bbox):
    return bbox[2] - bbox[0]

def measure_distance(p1, p2):
    return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5