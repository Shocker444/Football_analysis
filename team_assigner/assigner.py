from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import umap
from tqdm import tqdm
import numpy as np
import supervision as sv
import torch


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