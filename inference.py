'''
To run the code:
python inference.py --text "I'm looking for a sophisticated oxygen measurement device that can be used in a variety of settings, including labs and environmental monitoring. It should be capable of high-precision measurements in both gases and liquids."
--treshold 0.8
'''

import argparse
from Setup import MainDatabase

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_file_name", type=str,
                        default="/home/products_DB_with_id.json")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--treshold", type=float, default=0.5)
    parser.add_argument("--text", type=str)
    parser.add_argument("--number_results", type=int, default=59)
    parser.add_argument("--upload_file_name", type=str,
                        default="RAG_Products.json")
    parser.add_argument("--collection_name", type=str, default="my_collection")
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--embedding_model", type=str,
                        default="all-mpnet-base-v2")
    parser.add_argument(
        "--path", type=str, default="/home/zaven/Projects/se-llm/rag-development/llm_search_engine/chroma_data/chroma.sqlite3")
    parser.add_argument("--collection", default="all-mpnet-base-v2")
    args = parser.parse_args()

    data = MainDatabase(args,"embedding_info.json")
    data.evaluate()


