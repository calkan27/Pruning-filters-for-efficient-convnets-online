import torch
from sklearn.cluster import KMeans
import numpy as np

class Loss_Calculator(object):
    def __init__(self, num_clusters, model, device):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_seq = []
        self.num_clusters = num_clusters
        self.model = model
        self.device = device
        self.clusters = {}
        self.l1_weight = 0.001

    def calc_loss(self, output, target):
        loss = self.criterion(output, target)
        self.loss_seq.append(loss.item())
        # Add clustering loss
        cluster_loss = self.calculate_cluster_loss()
        l1_penalty = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
        total_loss = loss + self.l1_weight * l1_penalty + cluster_loss
        return total_loss

    def get_loss_log(self, length=100):
        if len(self.loss_seq) < length:
            length = len(self.loss_seq)
        return sum(self.loss_seq[-length:]) / length

    def calculate_cluster_loss(self):
        cluster_loss = 0
        for layer_index, (cluster_ids, cluster_centers) in self.clusters.items():
            module = self.model.features[layer_index]
            if isinstance(module, torch.nn.Conv2d):
                weights = module.weight.data.cpu().numpy()
                flattened_weights = weights.reshape(weights.shape[0], -1)  # Flatten the weights
                # Ensure cluster centers are reshaped correctly based on current weights dimensions
                valid_centers = [center[:flattened_weights.shape[1]] for center in cluster_centers if len(center) >= flattened_weights.shape[1]]

                # Summing the distances for each cluster center
                for idx, center in enumerate(valid_centers):
                    # Reshape center to ensure 1D and correct length
                    center = np.array(center).reshape(-1)[:flattened_weights.shape[1]]
                    # Calculate distances from all weights to this center
                    distances = np.linalg.norm(flattened_weights - center, axis=1)
                    # Sum distances of all weights assigned to this cluster
                    ids = cluster_ids[cluster_ids == idx]
                    current_distances = np.take(distances, ids)
                    # cluster_loss += np.sum(distances[cluster_ids == idx])
                    cluster_loss += np.sum(current_distances)

        return torch.tensor(cluster_loss, device=self.device, dtype=torch.float)
        
    def cluster_and_prune(self):
        # Extract conv layers from model
        conv_layers = [module for module in self.model.features if isinstance(module, torch.nn.Conv2d)]
        for i, conv in enumerate(conv_layers):
            weights = conv.weight.data.cpu().numpy()
            original_shape = weights.shape
            flattened_weights = weights.reshape(weights.shape[0], -1)  # Flatten the weights

            # Normalize the flattened weights
            norms = np.linalg.norm(flattened_weights, axis=1, keepdims=True)
            normalized_weights = flattened_weights / norms

            # Perform k-means clustering
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(normalized_weights)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_

            print(f"Layer {i}: Replacing {len(np.unique(labels))} clusters.")

            # Adjust centers to match the dimension of the flattened weights
            # Important: Only reshape centers if the dimensionality mismatch occurs
            if centers.shape[1] != flattened_weights.shape[1]:
                print(f"Adjusting center shape from {centers.shape} to {flattened_weights.shape[1]}")
                centers = centers[:, :flattened_weights.shape[1]]

            # Store clusters for loss calculation
            self.clusters[i] = (labels, centers)

            # Calculate cosine similarity and find the closest center for each filter
            similarities = np.dot(normalized_weights, centers.T)
            closest_center = np.argmax(similarities, axis=1)

            # Create a new set of weights with filters replaced by their closest cluster center
            new_weights = centers[closest_center]

            # Reshape weights to original shape and update the model
            new_weights = new_weights * norms  # Rescale to original norms
            new_weights = new_weights.reshape(original_shape)
            conv.weight.data = torch.tensor(new_weights, dtype=torch.float32, device=self.device)

    def prune_filters(self):
        print("Starting pruning of filters...")
        with torch.no_grad():
            self.cluster_and_prune()
        print("Finished pruning of filters.")

    def update_clusters(self):
        print("Updating clusters...")
        # Extract conv layers from model
        conv_layers = [module for module in self.model.features if isinstance(module, torch.nn.Conv2d)]
        for i, conv in enumerate(conv_layers):
            weights = conv.weight.data.cpu().numpy()
            original_shape = weights.shape
            flattened_weights = weights.reshape(weights.shape[0], -1)  # Flatten the weights

            # Normalize the flattened weights
            norms = np.linalg.norm(flattened_weights, axis=1, keepdims=True)
            normalized_weights = flattened_weights / norms

            # Perform k-means clustering
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(normalized_weights)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_

            # Adjust centers to match the dimension of the flattened weights
            # Important: Only reshape centers if the dimensionality mismatch occurs
            if centers.shape[1] != flattened_weights.shape[1]:
                print(f"Adjusting center shape from {centers.shape} to match flattened weights {flattened_weights.shape[1]}")
                centers = centers[:, :flattened_weights.shape[1]]

            # Store updated clusters for loss calculation and potential future use
            self.clusters[i] = (labels, centers)
        print("Clusters updated.")


            