from sklearn.cluster import KMeans

class TeamColorAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_player_color(self, frame, bbox):
        cropped = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        top_half = cropped[0:cropped.shape[0]//2, :, :]
        image_2d = top_half.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2)
        kmeans.fit(image_2d)

        labels = kmeans.labels_

        clustered_image = labels.reshape(top_half.shape[0], top_half.shape[1])
        corners = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corners), key=corners.count)

        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign(self, frame, tracks):

        player_colors = []

        for bbox in tracks:
            #bbox = players['bbox'] 
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
 
        kmean = KMeans(n_clusters=2)
        kmean.fit(player_colors)

        self.kmean = kmean

        self.team_colors[0] = tuple(kmean.cluster_centers_[0])
        self.team_colors[1] = tuple(kmean.cluster_centers_[1])

    def get_player_team(self, frame, player_bbox, player_id):

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmean.predict(player_color.reshape(1, -1))

        self.player_team_dict[player_id] = team_id

        return team_id
    
