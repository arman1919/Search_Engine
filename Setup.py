import chromadb
from chromadb.utils import embedding_functions
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import matplotlib.pyplot as plt
import numpy as np
from Methods import Chroma
from sentence_transformers import SentenceTransformer


class MainDatabase():
    def __init__(self, args,embedding_info_file):
        self.embedding_info_file = embedding_info_file
        self.args = args
        args.collection, self.check = self.create_collection()
        self.args = args
        self.data = self.load()
        self.prod = "PRODUCT CATEGORIES"
        self.target = "TARGET INDUSTRIES"


    def create_collection(self):

        chrome_client = chromadb.PersistentClient(path=self.args.path)


        # Create the collection with the embedding function

        self.args.collection = self.retrieve_collection()
        if self.args.collection != None:
            return self.args.collection, None
        else:
            try:
                embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.args.embedding_model,normalize_embeddings=True)
                # Manually extract necessary information from the embedding function
                embedding_info = {
                    "type": "SentenceTransformerEmbeddingFunction",
                    "model_name": self.args.embedding_model,
                    "normalize_embeddings": True
                }
                # Save information about the embedding function to a file
                with open(self.embedding_info_file, 'w') as f:
                    json.dump(embedding_info, f)

            except Exception:
                # Load the transformer model
                embedding_model = AutoModelForCausalLM.from_pretrained(
                    self.args.embedding_model, is_decoder=True)
                # Load the tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    self.args.embedding_model)

                # Check if the tokenizer has a padding token. If not, set it as the EOS token.
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                if tokenizer.pad_token is None:
                    tokenizer.add_special_tokens(
                        {'pad_token': f'[{self.args.tokenizer.eos_token}]'})

                # # Combine the transformer model and the tokenizer into a SentenceTransformer model
                # embedding_model = SentenceTransformer(
                #     modules=[transformer_model, tokenizer])
            self.args.collection = chrome_client.create_collection(name=self.args.collection_name, metadata={"hnsw:space": self.args.metric}, embedding_function=embedding_model)

            embedding_info = {
                    "type": "from_pretrained",
                    "model_name": self.args.embedding_model,
                    "is_decoder": True
                }

            # Save information about the embedding function to a file
            with open(self.embedding_info_file, 'w') as f:
                json.dump(embedding_info, f)

            return self.args.collection,not None

    def retrieve_collection(self):
        """
        Retrieve an existing collection with the specified parameters.

        Returns:
            Collection: Retrieved collection object.
        """
        chrome_client = chromadb.PersistentClient(path=self.args.path)

        try:
            # Attempt to delete the existing collection if it exists
            if chrome_client.get_collection(self.args.collection_name) != None:
                chrome_client.delete_collection(self.args.collection_name)
        except Exception:
            try:
                with open(self.embedding_info_file, 'r') as f:
                    embedding_info = json.load(f)

                # Create the collection with the embedding function parameters from the file
                if embedding_info['type'] == 'from_pretrained':
                    embedding_model = AutoModelForCausalLM.from_pretrained(
                        embedding_info['model_name'], is_decoder=embedding_info['is_decoder']
                    )
                else:
                    # Handle other types of embedding functions if needed
                    pass

                self.args.collection = chrome_client.create_collection(
                    name=self.args.collection_name,
                    metadata={"hnsw:space": self.args.metric},
                    embedding_function=embedding_model
                )

                return self.args.collection

            except Exception as e:
                print("Error creating collection:", e)
                return None

    def load(self):
        """
        Load data from a JSON file.

        Returns:
            dict: Loaded data.
        """
        if os.path.exists(self.args.load_file_name):
            with open(self.args.load_file_name, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data

    def add_to_collection(self):
        """
        Add data to the collection based on language.

        Returns:
            None
        """

        if self.check is not None:

            for prod in self.data:
                name = prod.get("Name", "")
                description = prod.get(
                    "Beschreibung" if self.args.language == "German" else "Description", "")
                applications = prod.get(
                    "ANWENDUNGEN" if self.args.language == "German" else "APPLICATIONS", "")
                prod_categories = prod.get(
                    "PRODUKTKATEGORIEN" if self.args.language == "German" else self.prod, "")
                target_industries = prod.get(
                    "ZIELBRANCHEN" if self.args.language == "German" else self.target, "")
                identify = prod.get('id')
                self.args.collection.add(
                    documents=[description],
                    metadatas={
                        "Description": ", ".join(description),
                        "APPLICATIONS": ", ".join(applications),
                        "PRODUCT CATEGORIES": ", ".join(prod_categories),
                        "TARGET INDUSTRIES": ", ".join(target_industries),
                        "id": identify
                    },
                    ids=[name]
                )

    def evaluate(self):

        self.add_to_collection()

        return Chroma(self.args).evaluate()