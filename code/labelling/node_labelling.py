import json
import pandas as pd

import braveblock

def get_resource_type(attr):

    """
    Function to get resource type of a node.

    Args:
        attr: Node attributes.
    Returns:
        Resource type of node.
    """

    try:
        attr = json.loads(attr)
        return attr['content_policy_type']
    except Exception as e:
        return None

def label_node_braveblock(row, adblocker):
    label = adblocker.check_network_urls(
        url = row['name'],
        source_url = row['top_level_url'],
        # Sometimes resource type can be undetermined(None) and check_network_urls requires type, but "" seems to work fine in those cases
        request_type = row['resource_type'] if row['resource_type'] is not None else ""
    )

    return label

def label_nodes(df):

    """
    Function to label nodes with filter lists.

    Args:
        df: DataFrame of nodes.
        filterlists: List of filter list names.
        filterlist_rules: Dictionary of filter lists and their rules.
    Returns:
        df_nodes: DataFrame of labelled nodes.
    """

    df_nodes = df[(df['type'] != 'Storage') & (df['type'] != 'Element')].copy()
    df_nodes['resource_type'] = df_nodes['attr'].apply(get_resource_type)

    adblocker = braveblock.Adblocker(
        include_easylist=True,  # Whether to include easylist rules. True by default
        include_easyprivacy=True,  # Whether to include easyprivacy rules. True by default
    )

    df_nodes['label'] = df_nodes.apply(label_node_braveblock, adblocker=adblocker, axis=1)
    df_nodes = df_nodes[['visit_id', 'name', 'top_level_url', 'label']]

    return df_nodes


def label_data(df):
    df_labelled = pd.DataFrame()

    try:
        df_nodes = df[df['graph_attr'] == "Node"]
        df_labelled = label_nodes(df_nodes)
    except Exception as e:
        LOGGER.warning("Error labelling:", exc_info=True)

    return df_labelled
