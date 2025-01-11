import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Union
import time

import pandas as pd
from tqdm import tqdm
from yaml import full_load
import multiprocessing
from itertools import repeat

import graph as gs
from graph.database import Database
import labelling as ls
from features.feature_extraction import extract_graph_features

from utils import return_none_if_fail
from logger import LOGGER

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.options.mode.chained_assignment = None   # Suppress pandas SettingWithCopyWarning warnings

def load_config_info(filename: str) -> dict:
    """Load features from features.yaml file
    :param filename: yaml file name containing feature names
    :return: dict of features to use.
    """
    with open(filename) as file:
        return full_load(file)


def extract_features(pdf: pd.DataFrame, networkx_graph, visit_id: str, config_info: dict, ldb) -> pd.DataFrame:
    """Getter to generate the features of each node in a graph.
    :param pdf: pandas df of nodes and edges in a graph.
    :param G: Graph object representation of the pdf.
    :return: dataframe of features per node in the graph
    """
    # Generate features for each node in our graph
    #ldb = leveldb.LevelDB(ldb_file)
    #ldb = plyvel.DB('/tmp/testdb/')

    df_features = extract_graph_features(pdf, networkx_graph, visit_id, ldb, config_info, lock)
    return df_features

@return_none_if_fail()
def find_setter_domain(setter: str) -> str:
    """Finds the domain from a setter
    :param setter: string setter value
    :return: string domain value
    """
    domain = gs.get_domain(setter)
    return domain

@return_none_if_fail(is_debug=True)
def find_domain(row: pd.Series) -> Union[str, None]:
    """Finds the domain of a node
    :param row: a row from the graph df representing a node.
    :return: string domain value or none if N/A
    """
    domain = None
    node_type = row['type']
    if (node_type == 'Document') or (node_type == 'Request') or (node_type == 'Script'):
        domain = gs.get_domain(row['name'])
    elif (node_type == 'Element'):
        return domain
    else:
        return row['domain']
    return domain

@return_none_if_fail()
def find_tld(top_level_url: str) -> Union[str, None]:
    """Finds the top level domain from a top level url
    :param top_level_url: string of the url
    :return: string domain value or none if N/A
    """
    if top_level_url:
        tld = gs.get_domain(top_level_url)
        return tld
    else:
        return None

def get_party(row: pd.Series) -> str:
    """Finds whether a storage node is first party or third party
    :param row: a row from the graph df representing a node.
    :return: string party (first | third | N/A)
    """
    if row['type'] == 'Storage':
        if row['domain'] and row['top_level_domain']:
            if row['domain'] == row['top_level_domain']:
                return 'first'
            else:
                return 'third'
    return "N/A"

def find_setters(df_all_storage_nodes: pd.DataFrame, df_http_cookie_nodes: pd.DataFrame, df_all_storage_edges: pd.DataFrame, df_http_cookie_edges: pd.DataFrame) -> pd.DataFrame:
    """Finds the nodes that first set each of the present cookies.
    :param row: a row from the graph df representing a node.
    :return: string domain value or none if N/A
    """

    df_setter_nodes = pd.DataFrame(columns=['visit_id', 'name', 'type', 'attr', 'top_level_url', 'domain', 'setter', 'setting_time_stamp'])

    try:

        df_storage_edges = pd.concat([df_all_storage_edges, df_http_cookie_edges])
        if len(df_storage_edges) > 0:
            # get all set events for http and js cookies
            df_storage_sets = df_storage_edges[(df_storage_edges['action'] == 'set')
                                | (df_storage_edges['action'] == 'set_js')]

            # find the initial setter nodes for each cookie
            df_setters = gs.get_original_cookie_setters(df_storage_sets)

            df_storage_nodes = pd.concat([df_all_storage_nodes, df_http_cookie_nodes])
            df_setter_nodes = df_storage_nodes.merge(df_setters, on=['visit_id', 'name'], how='outer')

    except Exception as e:
        LOGGER.warning("Error getting setter", exc_info=True)

    return df_setter_nodes


def build_graph(database: Database, visit_id: str, is_webgraph: bool) -> pd.DataFrame:
    """Read SQL data from crawler for a given visit_ID.
    :param visit_id: visit ID of a crawl URL.
    :param is_webgraph: compute additional info for WebGraph mode
    :return: Parsed information (nodes and edges) in pandas df.
    """
    # Read tables from DB and store as DataFrames
    df_requests, df_responses, df_redirects, call_stacks, javascript = database.website_from_visit_id(visit_id)

    # extract nodes and edges from all categories of interest as described in the paper
    df_js_nodes, df_js_edges = gs.build_html_components(javascript)
    df_request_nodes, df_request_edges = gs.build_request_components(df_requests, df_responses, df_redirects, call_stacks, is_webgraph)

    if is_webgraph:
        df_all_storage_nodes, df_all_storage_edges = gs.build_storage_components(javascript)
        df_http_cookie_nodes, df_http_cookie_edges = gs.build_http_cookie_components(df_request_edges, df_request_nodes)
        df_storage_node_setter = find_setters(df_all_storage_nodes, df_http_cookie_nodes, df_all_storage_edges, df_http_cookie_edges)
    else:
        df_all_storage_edges = pd.DataFrame()
        df_http_cookie_edges = pd.DataFrame()
        df_storage_node_setter = find_setters(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # Concatenate to get all nodes and edges
    df_request_nodes['domain'] = None
    df_all_nodes = pd.concat([df_js_nodes, df_request_nodes, df_storage_node_setter])
    df_all_nodes['domain'] = df_all_nodes.apply(find_domain, axis=1)
    df_all_nodes['top_level_domain'] = df_all_nodes['top_level_url'].apply(find_tld)
    df_all_nodes['setter_domain'] = df_all_nodes['setter'].apply(find_setter_domain)
    df_all_nodes = df_all_nodes.drop_duplicates()
    df_all_nodes['graph_attr'] = "Node"

    df_all_edges = pd.concat([df_js_edges, df_request_edges, df_all_storage_edges, df_http_cookie_edges])
    df_all_edges = df_all_edges.drop_duplicates()
    # 'top_level_url' can be missing from df
    if df_all_edges['top_level_url']:
        df_all_edges['top_level_domain'] = df_all_edges['top_level_url'].apply(find_tld)
    else:
        df_all_edges['top_level_domain'] = ""
    df_all_edges['graph_attr'] = "Edge"

    df_all_graph = pd.concat([df_all_nodes, df_all_edges])
    df_all_graph = df_all_graph.astype(
        {
            'type' : 'category',
            'response_status' : 'category'
        }
    )

    return df_all_graph

def build_graph_from_data(data, visit_id: str, is_webgraph: bool) -> pd.DataFrame:
    """Read SQL data from crawler for a given visit_ID.
    :param visit_id: visit ID of a crawl URL.
    :param is_webgraph: compute additional info for WebGraph mode
    :return: Parsed information (nodes and edges) in pandas df.


    """
    #print(f"Building graph for visit_id {visit_id}")

    # Read tables from DB and store as DataFrames
    df_requests, df_responses, df_redirects, call_stacks, javascript = data

    # extract nodes and edges from all categories of interest as described in the paper
    df_js_nodes, df_js_edges = gs.build_html_components(javascript)
    df_request_nodes, df_request_edges = gs.build_request_components(df_requests, df_responses, df_redirects, call_stacks, is_webgraph)

    if is_webgraph:
        df_all_storage_nodes, df_all_storage_edges = gs.build_storage_components(javascript)
        df_http_cookie_nodes, df_http_cookie_edges = gs.build_http_cookie_components(df_request_edges, df_request_nodes)
        df_storage_node_setter = find_setters(df_all_storage_nodes, df_http_cookie_nodes, df_all_storage_edges, df_http_cookie_edges)
    else:
        df_all_storage_edges = pd.DataFrame()
        df_http_cookie_edges = pd.DataFrame()
        df_storage_node_setter = find_setters(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


    # Concatenate to get all nodes and edges
    df_request_nodes['domain'] = None
    df_all_nodes = pd.concat([df_js_nodes, df_request_nodes, df_storage_node_setter])
    df_all_nodes['domain'] = df_all_nodes.apply(find_domain, axis=1)
    df_all_nodes['top_level_domain'] = df_all_nodes['top_level_url'].apply(find_tld)
    df_all_nodes['setter_domain'] = df_all_nodes['setter'].apply(find_setter_domain)
    df_all_nodes = df_all_nodes.drop_duplicates()
    df_all_nodes['graph_attr'] = "Node"

    df_all_edges = pd.concat([df_js_edges, df_request_edges, df_all_storage_edges, df_http_cookie_edges])
    df_all_edges = df_all_edges.drop_duplicates()

    # 'top_level_url' can be missing from df
    if 'top_level_url' in df_all_edges.columns:
        df_all_edges['top_level_domain'] = df_all_edges['top_level_url'].apply(find_tld)
    else:
        df_all_edges['top_level_domain'] = ""
    df_all_edges['graph_attr'] = "Edge"

    df_all_graph = pd.concat([df_all_nodes, df_all_edges])
    df_all_graph = df_all_graph.astype(
        {
            'type' : 'category',
            'response_status' : 'category'
        }
    )



    return df_all_graph

def apply_tasks(df: pd.DataFrame, visit_id: int, config_info: dict , ldb_file: Path, output_dir: Path, overwrite: bool):

    """ Sequence of tasks to apply on each website crawled.
    :param df: the graph data (nodes and edges) in pandas df.
    :param visit_id: visit ID of a crawl URL.
    :param config_info: dictionary containing features to use.
    :param ldb_file: path to ldb file.
    :param output_dir: path to the output directory.
    :param overwrite: set True to overwrite the content of the output directory.
    """

    # Build the graph
    LOGGER.info("%s %d %d", df.iloc[0]['top_level_url'], visit_id, len(df))
    graph_columns = config_info['graph_columns']
    feature_columns = config_info['feature_columns']
    label_columns = config_info['label_columns']

    try:
        start = time.time()

        # export the graph dataframe to csv file
        graph_path = output_dir / "graph.csv"
        if overwrite or not graph_path.is_file():
            df.reindex(columns=graph_columns).to_csv(str(graph_path))
        else:
            df.reindex(columns=graph_columns).to_csv(str(graph_path), mode='a', header=False)

        # building networkx_graph
        networkx_graph = gs.build_networkx_graph(df)

        # extracting features from graph data and graph structure
        df_features = extract_features(df, networkx_graph, visit_id, config_info, ldb_file)
        features_path = output_dir / "features.csv"

        # export the features to csv file
        if overwrite or not features_path.is_file():
            df_features.reindex(columns=feature_columns).to_csv(str(features_path))
        else:
            df_features.reindex(columns=feature_columns).to_csv(str(features_path), mode='a', header=False)
        end = time.time()
        LOGGER.info("Extracted features: %d", end-start)

        #Label data
        df_labelled = ls.label_data(df)
        if len(df_labelled) > 0:
            labels_path = output_dir / "labelled.csv"
            if overwrite or not labels_path.is_file():
                df_labelled.reindex(columns=label_columns).to_csv(str(labels_path))
            else:
                df_labelled.reindex(columns=label_columns).to_csv(str(labels_path), mode='a', header=False)

    except Exception as e:
        LOGGER.warning("Errored in pipeline", exc_info=True)

def apply_tasks_multi(df: pd.DataFrame, visit_id: int, config_info: dict , ldb_file):

    """ Sequence of tasks to apply on each website crawled.
    :param df: the graph data (nodes and edges) in pandas df.
    :param visit_id: visit ID of a crawl URL.
    :param config_info: dictionary containing features to use.
    :param ldb_file: path to ldb file.
    """

    # Build the graph
    LOGGER.info("%s %d %d", df.iloc[0]['top_level_url'], visit_id, len(df))


    try:
        start = time.time()

        # building networkx_graph
        networkx_graph = gs.build_networkx_graph(df)


        # extracting features from graph data and graph structure
        df_features = extract_features(df, networkx_graph, visit_id, config_info, ldb_file)

        end = time.time()
        LOGGER.info("Extracted features: %d", end-start)

        #Label data
        df_labelled = ls.label_data(df)

        return (df_features, df_labelled)

    except Exception as e:
        LOGGER.warning("Errored in pipeline", exc_info=True)

def validate_config(config_info: Dict[str, Any], mode: str) -> None:
    """ Validate the configuration.
    :param config_info: dictionary containing features to use.
    :param mode: computation mode
    """
    if mode == "adgraph":
        if "dataflow" in config_info["features_to_extract"]:
            raise Exception("Error: 'dataflow' must not be used in 'adgraph' mode, please remove it from 'features_to_extract'.")


def graph_to_file(graph_df, output_dir, config_info):
    graph_columns = config_info['graph_columns']
    graph_path = output_dir / "graph.csv"

    # export the graph dataframe to csv file
    if not graph_path.is_file():
        graph_df.reindex(columns=graph_columns).to_csv(str(graph_path))
    else:
        graph_df.reindex(columns=graph_columns).to_csv(str(graph_path), mode='a', header=False)

def features_labels_to_file(df_features, df_labels, output_dir, config_info):
    feature_columns = config_info['feature_columns']
    label_columns = config_info['label_columns']

    # Export the features to csv file
    features_path = output_dir / "features.csv"
    if not features_path.is_file():
        df_features.reindex(columns=feature_columns).to_csv(str(features_path))
    else:
        df_features.reindex(columns=feature_columns).to_csv(str(features_path), mode='a', header=False)


    # Export labels
    if len(df_labels) > 0:
        labels_path = output_dir / "labelled.csv"
        if not labels_path.is_file():
            df_labels.reindex(columns=label_columns).to_csv(str(labels_path))
        else:
            df_labels.reindex(columns=label_columns).to_csv(str(labels_path), mode='a', header=False)

def init(l):
    """ This method is used to initialize a multiprocessing lock as a global variable inside a new process.
    It's a weird case where normal Locks can't be directly passed into new processes when using multiprocessing.Pool
    source: https://stackoverflow.com/questions/25557686/python-sharing-a-lock-between-processes/25558333#25558333

    Args:
        l: Multiprocessing Lock

    Returns:

    """
    global lock
    lock = l

def pipeline(db_file: Path, ldb_file: Path, features_file: Path, output_dir: Path, mode: str, ns):

    """ Graph processing and labeling pipeline
    :param db_file: the graph data (nodes and edges) in pandas df.
    :param visit_id: visit ID of a crawl URL.
    :param config_info: dictionary containing features to use.
    :param ldb_file: path to ldb file.
    :param output_dir: path to the output directory.
    :param mode: computation mode
    :param overwrite: set True to overwrite the content of the output directory.
    """

    number_failures = 0

    config_info = load_config_info(features_file)

    validate_config(config_info, mode)

    output_dir.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 100
    THREADS = ns.threads

    with Database(db_file) as database:

        # read site visits
        try:
            sites_visits = database.sites_visits()
        except Exception as e:
            tqdm.write(f"Problem reading the sites_visits: {e}")
            exit()

        batch_nr = 0
        with tqdm(total=sites_visits.shape[0]) as pbar:
            while True:
                website_data_batch = []     # Inputs from database for current batch
                visit_id_batch = []
                start_idx = BATCH_SIZE * batch_nr
                end_idx = min(start_idx + BATCH_SIZE, sites_visits.shape[0])    # Stop after dataframe rows are exhausted

                # SQLite does not play well with multiprocessing so I'll query a batch data beforehand
                for row_idx in range(start_idx, end_idx):
                    visit_id = sites_visits['visit_id'][row_idx]
                    data = database.website_from_visit_id(visit_id) # returns (df_requests, df_responses, df_redirects, call_stacks, javascript)
                    website_data_batch.append(data)
                    visit_id_batch.append(visit_id)

                # Create process pool with a lock for leveldb access management
                l = multiprocessing.Lock()
                with multiprocessing.Pool(THREADS, initializer=init, initargs=(l,)) as p:
                    print(f"Building graphs for batch={batch_nr}")
                    graphs = p.starmap(build_graph_from_data, zip(website_data_batch, visit_id_batch, repeat(mode=="webgraph")))

                    print(f"Extracting features and labelling batch={batch_nr}")
                    features_labels = p.starmap(apply_tasks_multi, zip(graphs, visit_id_batch, repeat(config_info), repeat(ldb_file)))

                print(f"Finished batch={batch_nr}")

                # Write batch results to file
                for graph_df in graphs:
                    graph_to_file(graph_df, output_dir, config_info)


                for result in features_labels:
                    # Sometimes feature extraction returns None
                    if result is None:
                        continue

                    df_features, df_labelled = result
                    features_labels_to_file(df_features, df_labelled, output_dir, config_info)


                with open("status.txt", "a") as f:
                    f.write(f"Finished processing batch, {end_idx}/{sites_visits.shape[0]}")

                pbar.update(end_idx-start_idx)
                batch_nr += 1

                # All sites have been processed
                if end_idx >= sites_visits.shape[0]:
                    break


    percent = (number_failures/len(sites_visits))*10
    LOGGER.info(f"Fail: {number_failures}, Total: {len(sites_visits)}, Percentage:{percent} %s", str(db_file))


def main(program: str, args: List[str]):

    parser = argparse.ArgumentParser(prog=program, description="Run a classification pipeline.")
    parser.add_argument(
        "--input-db",
        type=Path,
        help="Input SQLite database.",
        default=Path("crawl-data.sqlite")
    )
    parser.add_argument(
        "--ldb",
        type=str,
        help="Input LDB.",
        default="content.ldb"
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Features.",
        default=Path("features.yaml")
    )
    parser.add_argument(
        "--filters",
        type=Path,
        help="Filters directory.",
        default=Path("filterlists")
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Directory to output the results.",
        default=Path("out")
    )

    parser.add_argument(
        "--mode",
        choices=["webgraph", "adgraph"],
        help="Computation mode",
        default="webgraph"
    )

    parser.add_argument(
        "--threads",
        type=int,
        help="Number of threads to use for multiprocessing.",
        default=1
    )


    ns= parser.parse_args(args)

    pipeline(ns.input_db, ns.ldb, ns.features, ns.out, ns.mode, ns)


if __name__ == "__main__":
    main(sys.argv[0], sys.argv[1:])

