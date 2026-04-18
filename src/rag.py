import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

# Find where our engagement_strategies.csv file is located
CURRENT_FOLDER = os.path.dirname(__file__)
PARENT_FOLDER = os.path.dirname(CURRENT_FOLDER)
DATA_FILE_PATH = os.path.join(PARENT_FOLDER, 'data', 'engagement_strategies.csv')

class StrategyRAG:
    """
    RAG means Retrieval-Augmented Generation. 
    This class loads our strategies and helps us search for the best one.
    """
    def __init__(self):
        # Step 1: Load a free, local AI model that understand meanings of sentences (embeddings)
        print("Loading AI embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_database = None
        
        # Step 2: Read our strategies from the CSV file and save them in our database
        self.load_data_into_database()

    def load_data_into_database(self):
        """ This reads the CSV file and creates a searchable AI database. """
        # Check if file exists
        if os.path.exists(DATA_FILE_PATH) == False:
            print(f"Error: Could not find {DATA_FILE_PATH}")
            return
            
        # Read the CSV with pandas
        dataframe = pd.read_csv(DATA_FILE_PATH)
        list_of_documents = []
        
        # Go through each row one by one
        for index, row in dataframe.iterrows():
            # Combine all the text so the AI can understand it
            text_for_ai = f"Player Match: {row['target_audience']}. Problem: {row['scenario']}. Solution: {row['recommended_action']}. Expected Result: {row['expected_outcome']}."
            
            # Create a document object
            new_doc = Document(
                page_content=text_for_ai,
                metadata={
                    "target_audience": row["target_audience"],
                    "scenario": row["scenario"],
                    "action": row["recommended_action"],
                    "expected_outcome": row["expected_outcome"]
                }
            )
            # Add it to our list
            list_of_documents.append(new_doc)
            
        # Create a FAISS database out of our documents
        self.vector_database = FAISS.from_documents(list_of_documents, self.embeddings)
        print("Successfully loaded strategies into FAISS database!")

    def retrieve_strategies(self, player_profile_text, number_of_results=2):
        """ This searches the database for the best strategy for a specific player. """
        if self.vector_database == None:
            return [{"error": "Database is empty."}]
            
        # AI magically finds the most similar strategies to the player profile
        matching_documents = self.vector_database.similarity_search(player_profile_text, k=number_of_results)
        
        # Make a simple list of dictionaries to return
        found_strategies = []
        for doc in matching_documents:
            strategy_info = {
                "audience": doc.metadata["target_audience"],
                "scenario": doc.metadata["scenario"],
                "action": doc.metadata["action"],
                "outcome": doc.metadata["expected_outcome"]
            }
            found_strategies.append(strategy_info)
            
        return found_strategies