# Search engine

## Product search using vector database and LLM

## The program consists of 2 parts 

# Part 1

## Filtering products on request from a large-scale database using Chroma

# Part 2

# From the filtered part using LLAMA 7b extraction of more specific products




# 1 Chroma Database

## Overview

Chroma Database is a Python project designed to set up a database for storing and querying product information. It provides functionality for creating collections within the database, adding data to these collections, performing similarity calculations based on user queries, and filtering results based on specified criteria.

## Files

- **Setup.py**: Contains the `MainDatabase` class responsible for setting up the main database, creating collections, loading data, adding data to collections, and evaluating the system.

- **Methods.py**: Defines the `Chroma` class, which handles methods related to calculating similarity, filtering collections, getting datasets, and evaluating the system.

- **inference.py**: Entry point for running the code. Parses command-line arguments using `argparse`, instantiates the `MainDatabase` class, and initiates the evaluation process.

## Usage

To use Chroma Database, follow these steps:

1. Ensure you have Python installed on your system.

2. Install the required dependencies by running:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your data:
   - Ensure your data is in a JSON format with specific fields like Name, Description, Applications, Product Categories, and Target Industries.
   - Optionally, provide a SQLite database containing your data.

4. Run the code:
   ```
   python inference.py --text "<query_text>" --threshold <threshold_value>
   ```

   Replace `<query_text>` with the text of your query and `<threshold_value>` with the desired threshold for similarity calculations.

5. View the results:
   - The program will output filtered product information based on the query and threshold specified.
   - Results will be saved in a JSON file.

## Additional Notes

- The project includes functionalities for creating collections, loading data, calculating similarities, and filtering results. You can extend it further based on your specific requirements.

- Ensure you have the necessary permissions and access to any databases or files referenced in the code.




# 2 Product Search using LLM

This script utilizes a Large Language Model (LLM) to search for products based on their descriptions. The process involves loading a pre-trained LLM model, creating a prompt for product search queries, and extracting product IDs from the model's response.


## Usage

1. **Prepare Data**: Ensure your product data is stored in a JSON file (`products_DB_(EN).json`) with the following structure:

```json
[
    {
        "Name": "Product Name 1",
        "Description": "Product Description 1",
        "APPLICATIONS": ["App1", "App2"],
        "PRODUCT CATEGORIES": ["Category1", "Category2"],
        "TARGET INDUSTRIES": ["Industry1", "Industry2"]
    },
    {
        "Name": "Product Name 2",
        "Description": "Product Description 2",
        "APPLICATIONS": ["App3", "App4"],
        "PRODUCT CATEGORIES": ["Category3", "Category4"],
        "TARGET INDUSTRIES": ["Industry3", "Industry4"]
    },
    ...
]
```


2. **Interact with the Script**: The script will prompt you with a user query. Provide a description of the product you are searching for. The script will return a list of product IDs that match the query.

## Description

- `extract_list(text)`: Extracts a list of identifiers from the LLM response.
- `enumerate_list(products)`: Enumerates the list of products and assigns IDs.
- `dicts_to_str(products)`: Converts the list of product dictionaries into a string representation.
- `creat_db(products)`: Creates a database for product search.
- `pipeline`: Loads the LLM model and creates a pipeline for text generation.
- `messages`: Defines a prompt for the LLM model.
- `prompt`: Applies the chat template to create a prompt for the LLM model.
- `terminators`: Defines terminators for the LLM model.
- `outputs`: Generates a response from the LLM model based on the prompt.
- `Product_IDs`: Extracts product IDs from the LLM response.

## Output

The script will output the found products along with the LLM's response.