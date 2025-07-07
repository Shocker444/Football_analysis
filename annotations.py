import supervision as sv
from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()

ellipse_annotator = sv.EllipseAnnotator(color = sv.ColorPalette.from_hex(['#00ff1b', '#0ed0ff', '#ffae00']),
                                thickness=2)

triangle_annotator = sv.TriangleAnnotator(color = sv.Color.from_hex('#ff0700'),
                                        base=20, height=18)


label_annotator = sv.LabelAnnotator(
        color = sv.ColorPalette.from_hex(['#00ff1b', '#0ed0ff', '#ffae00']),
        text_color = sv.Color.from_hex("#000000"),
        text_position = sv.Position.BOTTOM_CENTER
    )

vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex('#FF1493'),
                                      radius=8)

edge_annotator = sv.EdgeAnnotator(color=sv.Color.from_hex('#00BFFF'),
                                  thickness=2,
                                  edges=CONFIG.edges)