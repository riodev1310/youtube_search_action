import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import pairwise_distances

# helper function
def returnSearchResultIndexes(
    query: str, 
    df: pl.LazyFrame, 
    model: SentenceTransformer
) -> np.ndarray:
    """
    Function to return indexes of top search results
    """
    # embed query
    query_embedding = model.encode(query).reshape(1, -1)  # shape (1, 384)

    # collect embedding columns once (columns[4:] are embeddings)
    embedding_df = df.select(df.columns[4:]).collect()
    embeddings = embedding_df.to_numpy()  # shape (n_videos, 768)

    # split into two parts (original logic: 4:388 và 388:)
    emb1 = embeddings[:, :384]   # first 384 dims
    emb2 = embeddings[:, 384:]   # remaining 384 dims

    # compute manhattan distances
    dist1 = pairwise_distances(emb1, query_embedding, metric='manhattan').flatten()
    dist2 = pairwise_distances(emb2, query_embedding, metric='manhattan').flatten()

    dist_arr = dist1 + dist2

    # search parameters
    threshold = 40
    top_k = 5

    # filter by threshold
    idx_below_threshold = np.argwhere(dist_arr < threshold).flatten()

    if len(idx_below_threshold) == 0:
        return np.array([], dtype=int)

    # sort and take top k
    idx_sorted = np.argsort(dist_arr[idx_below_threshold])[:top_k]
    return idx_below_threshold[idx_sorted]