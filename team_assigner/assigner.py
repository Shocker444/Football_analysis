from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import umap
from tqdm import tqdm
import numpy as np
import supervision as sv
import torch

'''class TeamColorAssigner:
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

        return team_id'''

class TeamAssigner:
    def __init__(self, device='cpu', batch_size=32, siglip_path = 'google/siglip-base-patch16-224'):

        self.device= device
        self.batch_size = batch_size
        self.embedding_model = SiglipVisionModel.from_pretrained(siglip_path).to(device)
        self.embedding_processor = AutoProcessor.from_pretrained(siglip_path)
        self.reducer = umap.UMAP(n_components=3)
        self.clustering_mod = KMeans(n_clusters=2)

    def extract_features(self, crops, show_progress=True):
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batch_ds = chunked(crops, self.batch_size)

        data = []

        disable = False if show_progress else True
        with torch.no_grad():
            for batch in tqdm(batch_ds, desc = 'generating embeddings', total=int(len(crops)/self.batch_size), disable=disable):
                inputs = self.embedding_processor(images=batch, return_tensors='pt')
                outputs = self.embedding_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)
        
        data = np.concatenate(data)

        return data

    def fit(self, crops):
        embeddings = self.extract_features(crops)
        reduced_umap = self.reducer.fit_transform(embeddings)
        self.clustering_mod.fit(reduced_umap)
        
    def predict(self, crops):
        embeddings = self.extract_features(crops, show_progress=False)
        reduced_umap = self.reducer.transform(embeddings)
        predictions = self.clustering_mod.predict(reduced_umap)
        return predictions