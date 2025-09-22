import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

class FastClusterSearch(object):
    def __init__(self, mode="cos_sim"):
        """
        Initialize clustering and search tool.
        Args:
            mode (str): "l2" or "cos_sim"
        """
        mode = mode.lower()
        if mode == "l2":
            self.metric = "l2"
        elif mode == "cos_sim":
            self.metric = "cosine"
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    
    # @staticmethod
    def private_cluster(self, public_clusters, priv_embedding, sigma, R):
        """
        Args:
            public_clusters (List[Dict]): each with keys:
                - 'center': np.ndarray of shape (D,)
                - 'size': float
            priv_embedding (List[Dict]): each with keys:
                - 'embedding': np.ndarray of shape (D,)
                - 'sample probablity': float
            sigma (float): noise scale
            R (float): clipping l2-norm
        Returns:
            List[Dict]: same keys as input clusters: 'center', 'size'
        """

        # -------- Parse public clusters --------
        public_centers = np.stack([c['center'] for c in public_clusters], axis=0)  # (M, D)
        d = public_centers.shape[1]
        public_cluster_size = np.array([c['size'] for c in public_clusters], dtype=float)  # (M,)

        # Running sums for centers (center * size)
        embedding_sum = public_centers * public_cluster_size[:, None]
        cluster_size = public_cluster_size.copy()

        # -------- Parse private pool (keep raw list separate to avoid variable overwrite bugs) --------
        priv_pool_embeddings = np.stack([p['embedding'] for p in priv_embedding], axis=0)  # (N, D)
        probs = np.array([p['sample_prob'] for p in priv_embedding], dtype=float) # (N,)

        if (not np.isfinite(probs).all()) or (probs < 0).any() or (probs > 1).any():
            raise ValueError("Sampling probabilities must be in [0, 1] and finite.")

        # -------- Bernoulli sampling --------
        mask = np.random.binomial(n=1, p=probs).astype(bool)  # True 表示保留
        selected_priv = priv_pool_embeddings[mask]  # (K, D), K <= N

        if selected_priv.shape[0] == 0:
            return public_clusters

        # -------- Choose nearest public cluster --------
        # NOTE: assumes self.metric is defined ('L2', 'cos_sim', etc.)
        nn = NearestNeighbors(n_neighbors=1, metric=self.metric, algorithm="brute", n_jobs=-1)
        nn.fit(public_centers)
        _, nearest_ids = nn.kneighbors(selected_priv)  # (K, 1)
        nearest_ids = nearest_ids.flatten()

        for vec, cid in zip(selected_priv, nearest_ids):
            v_norm = np.linalg.norm(vec)
            if v_norm > 0:
                vec = vec * min(1.0, R / v_norm)
            embedding_sum[cid] += vec
            cluster_size[cid] += 1.0

        out = []
        for i in range(len(public_clusters)):
            n_i = max(cluster_size[i], 1.0)
            mean_i = embedding_sum[i] / n_i
            
            n_public = max(public_cluster_size[i], 1.0)

            noisy_size = cluster_size[i] + np.random.normal(0, sigma)
            noisy_size = float(max(noisy_size, 0.0))

            center_noise = (2.0 * R / (n_public + noisy_size)) * np.random.normal(0, sigma, size=d)
            noisy_center = mean_i + center_noise
            
            out.append({
                'center': noisy_center,
                'size': noisy_size
            })

        return out

    def K_cluster(self, embeddings, num_clusters):
        if len(embeddings) < num_clusters:
            return []

        # normed_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        kmeans = KMeans(n_clusters=num_clusters, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(embeddings)
        #labels = kmeans.fit_predict(normed_embeddings)
        centers = kmeans.cluster_centers_

        clusters = []
        for k in range(num_clusters):
            mask = (labels == k)
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]  # use position in embeddings
            # cluster_points = embeddings[mask]
            # center = np.mean(cluster_points, axis=0)
            center = centers[k]
            clusters.append({
                'indices': idx.tolist(),
                'center': center
            })
        return clusters
    

    def K_cluster_new(self, embeddings, num_clusters):
        X = np.asarray(embeddings)
        n = X.shape[0]
        if n == 0: return []
        if n < num_clusters: num_clusters = n

        if self.metric == "cos_sim":
            eps = 1e-12
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.maximum(norms, eps)


        kmeans = KMeans(n_clusters=num_clusters, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_

        if self.metric == "cos_sim":
            c_norms = np.linalg.norm(centers, axis=1, keepdims=True)
            centers = centers / np.maximum(c_norms, 1e-12)

        clusters = []
        for k in range(num_clusters):
            size = int(np.sum(labels == k))
            if size == 0:
                continue
            clusters.append({
                'center': centers[k],      # np.ndarray of shape (D,)
                'size': size       # int
            })
        return clusters


    def search(self, syn_embedding, clusters):
        """
        For each private sample, calculate votes

        Args:
            syn_embedding (np.ndarray): shape (M, D)
            clusters (List[Dict]): each with keys:
                - 'center': np.ndarray of shape (D,)
                - 'size': int

        Returns:
            List[List[int]]: length N, each entry is list of synthetic sample IDs
        """
        M = syn_embedding.shape[0]
        if M == 0 or len(clusters) == 0:
            return np.zeros(M, dtype=int)

        centers = np.stack([cluster['center'] for cluster in clusters])    # shape: (num_clusters, D)
        sizes = np.array([int(c['size']) for c in clusters], dtype=np.int64)

        if self.metric == "cos_sim":
            eps = 1e-12
            norms = np.linalg.norm(syn_embedding, axis=1, keepdims=True)
            syn_embedding = syn_embedding / np.maximum(norms, eps)

        nn = NearestNeighbors(n_neighbors=1, metric=self.metric, algorithm='brute', n_jobs=-1)
        nn.fit(syn_embedding)
        _, nn_ids = nn.kneighbors(centers, n_neighbors=1)                    # (K,1)
        nn_ids = nn_ids.reshape(-1)

        votes = np.zeros(M, dtype=np.int64)

        # expanded_sizes = np.repeat(sizes, 2)  # 原始 sizes (K,) → 扩展后 (6K,)
        # np.add.at(votes, nn_ids, expanded_sizes)
        np.add.at(votes, nn_ids, sizes)

        return votes.astype(int)
    
