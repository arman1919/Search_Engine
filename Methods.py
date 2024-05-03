from chromadb.utils import embedding_functions
import json
from transformers import AutoModelForCausalLM
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import sqlite3
import re


class Chroma:
    def __init__(self, args):
        """
        Initialize Chroma object with provided arguments.

        Args:
            args (Namespace): Parsed command-line arguments.
        """
        self.args = args
        self.data = self.load()
        self.filtered_data = {}
        self.filtered_ids = {}
        self.prod = "PRODUCT CATEGORIES"
        self.target = "TARGET INDUSTRIES"

    def load(self):
        # Connect to the SQLite database
        conn = sqlite3.connect(f"{self.args.path}/chroma.sqlite3")

        # Create a cursor object
        cur = conn.cursor()

        # Query the embedding_queue table
        cur.execute("SELECT id, vector, metadata FROM embeddings_queue")

        # Fetch all rows from the query
        rows = cur.fetchall()

        # Close the connection
        conn.close()

        # Structure the data
        data = [{'id': row[0], 'vector': row[1], 'metadata': row[2]}
                for row in rows]

        #print(f" This is the embedding vector: {data[0]['vector']}")

        return data

    def calculate_similarity(self):
        """
        Calculate similarity between text and collection.

        Returns:
            dict: Calculated similarities.
        """
        collect = self.args.collection.query(
            query_texts=[self.args.text],
            n_results=self.args.number_results,
            include=['distances', 'metadatas', 'documents']
        )

        return collect

    def filter_collection(self):
        """
        Filter products based on distance threshold and metric.

        Returns:
            None
        """
        for product_id, distance in zip(self.filtered_data['ids'][0], self.filtered_data['distances'][0]):
            if (distance > self.args.treshold and self.args.metric == "cosine") or (self.args.metric != "cosine" and distance < self.args.treshold):
                self.filtered_ids[product_id] = distance

    def get_dataset(self):
        """
        Get dataset based on language and filtered IDs.

        Returns:
            None
        """
        pattern = r'"(.*?)\"'

        filtered_products = []
        for prod in self.data:
            if prod['id'] in self.filtered_ids.keys():

                matches = re.findall(pattern, prod['metadata'])

                filtered_products.append({
                    "Name": prod.get('id', ""),
                    "Description": matches[1],
                    "APPLICATIONS": matches[3],
                    "PRODUCT CATEGORIES": matches[5],
                    "TARGET INDUSTRIES": matches[7]
                })
        if os.path.exists("/home/zaven/Projects/se-llm/rag-development/llm_search_engine/RAG_Products_(English).json"):
            os.remove("/home/zaven/Projects/se-llm/rag-development/llm_search_engine/RAG_Products_(English).json")
        with open(f"{self.args.upload_file_name.split('.')[0]}_({self.args.language}).{self.args.upload_file_name.split('.')[1]}", "w") as f:
            json.dump(filtered_products, f, indent=4)

    def evaluate(self):
        """
        Evaluate the system by creating a collection, adding data, calculating similarity,
        filtering collection, getting dataset, and saving results.

        Returns:
            None
        """

        self.filtered_data = self.calculate_similarity()
        self.filter_collection()
        self.get_dataset()
        self.cache_clear()

    def DistancesPlotBar(self):
        distances = self.get_scores()
        indices = list(range(1, len(distances) + 1))  # Generate indices from 1 to the number of distances
        np.random.shuffle(indices)  # Shuffle the indices randomly

        fig, ax = plt.subplots(figsize=(20, 10))  # Set the figure size
        
        # Define shades of blue
        colors = ['#1f77b4', '#4a90e2', '#6ab0f3', '#9acfea']  # You can add more shades if needed
        
        threshold = 0.5
        
        # Plot bars with different colors
        for i in range(0, len(distances), 4):
            group_distances = distances[i:i+4]
            group_indices = indices[i:i+4]
            group_colors = colors[:len(group_distances)]
            ax.bar(group_indices, group_distances, color=group_colors)
        
        ax.set_title("Distances between input sentence and products")
        ax.set_xlabel("Product ID")
        ax.set_ylabel("Distance")
        ax.set_xticks(np.arange(1, 61))  # Set x-axis ticks to show product IDs from 1 to 60 with intervals of 5

        if threshold is not None:
            ax.axhline(y=threshold, color='k', linestyle='--', label=f'Threshold = {threshold}')
            ax.legend()  # Add legend for the threshold line

        plt.ylim(0, max(distances) + 0.1)  # Set y-axis limits based on the maximum distance plus a small buffer

        plt.show()
    
    def get_scores(self):
        return {key: value for key, value in zip(self.filtered_ids.keys(), self.filtered_data["distances"][0])}

    def plot(self):
        scores = self.get_scores()
        product_names = np.arange(1, len(scores.values())+1, step=1)
        scores_values = list(scores.values())

        plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
        plt.plot(product_names, scores_values,
                 color='green', marker='o', linestyle='-')
        plt.ylabel('Scores')
        plt.xlabel('Products')
        plt.title('Product Scores')
        plt.xticks(product_names)
        plt.grid()
        plt.savefig("my_plot.png")  # Save the plot before showing it
        plt.show()

    def Plottt(self):
        # Create a plot
        _, ax = plt.subplots()

        scores = self.get_scores()

        ids_names = list(scores.keys())

        score = {key: value for key, value in enumerate(ids_names)}
        # Plot each ID and name as text
        for i, (key, value) in enumerate(score.items()):
            ax.text(0.1, 1 - i * 0.1, f"{key}: {value}",
                    transform=ax.transAxes, fontsize=12)

        # Remove axes
        ax.axis("off")

        # Adjust layout
        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig("my_plot_ids.png", dpi=300)

        # Show the plot (optional)
        plt.show()

    def cache_clear(self):
        torch.cuda.empty_cache()