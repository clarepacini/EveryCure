#!/usr/bin/env python3
import torch
from torch_geometric.data import Data

import pandas as pd




def load_data(node_file, edge_file):
    """
    Load nodes and edges from CSV files.
    """
      # Load the node features from CSV
    node_features_df = pd.read_csv(node_features_file)
    node_ids = node_features_df['id']
    #this is the part of the code that will not currently work as need to parse the data
   # node_features = torch.tensor(node_features_df.drop(columns=['id']).values, dtype=torch.float)
    # Generate random node features as a place holder because node feature parsing to be added
    node_features = torch.rand((num_nodes, 2))  # 2D features for each node
    # Load the edge data from CSV
    edge_data_df = pd.read_csv(edges_file)
    edge_index = torch.tensor(edge_data_df[['source', 'target']].values.T, dtype=torch.long)

    # Extract unique edge types from the 'type' column
    edge_types = edge_data_df['type'].values
    unique_edge_types = sorted(set(edge_types))  # Sort to ensure consistent mapping

    # Create a mapping of edge types to integer values
    edge_type_mapping = {etype: idx for idx, etype in enumerate(unique_edge_types)}

    # Convert edge types to their corresponding numeric values
    edge_attr = torch.tensor([edge_type_mapping[et] for et in edge_types], dtype=torch.long)

 
    
    return node_features, node_ids,edge_index, edge_attr

 

def build_knowledge_graph(node_file, edge_file, output_file):
    """
    Build a knowledge graph from node and edge CSV files and save it to a .pt file.
    """
    node_features, node_ids, edge_index, edge_attr = load_data(node_file, edge_file)
       # Create a torch_geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    # Save the graph object to a file
    torch.save(data, output_file)
    print(f"Graph saved to {output_file}")
  
    # Save node IDs separately when saving the graph

    node_ids.to_csv("node_ids.csv", index=False)
 
    
   


  
