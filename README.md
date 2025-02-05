# WebGraph

## Update: January 2025

I updated WebGraph as part of my Semester Project at EPFL SPRING Lab. Main contributions:
- Graph building and feature extraction (`run.py`)
  - Ran performance profiling on graph building and feature extraction. Determined that the two main slowdowns were
URL checks against ad-blocker filterlist and feature extraction relying on networkx (implemented in pure python).
  - Replaced the previous implementation of checking URLs against ad-blocker lists (that was slow and broken as of dec.2024) 
with the `braveblock` library written in rust that utilizes EasyList and EasyPrivacy as the blocklists.
  - Added multiprocessing to graph building and feature extraction. Use flag `--threads [int]` to set the number of 
parallel threads to use.
  - Fixed a handful on minor bugs that caused some graph builds and feature extractions to fail.
- Machine Learning (`classification/classify.py`)
  - Added optional SMOTE upsampling for minority class. Use flag `--resample` to include upsampling.
  - Fixed bug with boolean command line flags always evaluating to True. New usage example for boolean flags: instead of 
`--save True` use `--save` for True and remove flag for False. 

### Notes on WebGraph installation
Installing the dependencies in `requirements.txt` will fail with newer versions of python. Our investigation showed that the
most up-to-date version that WebGraph works with is `python 3.9`. 

#### Installing old python
```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.9 python3.9-dev python3.9-venv
```

#### Running WebGraph
The newer versions of OpenWPM will produce a SQLite file that is named `crawl-data.sqlite` instead of `crawl-db.sqlite`.
When using newer versions of OpenWPM the usage of WebGraph `run.py` is as follows:
```
python3 ~/WebGraph/code/run.py --input-db ~/datadir/crawl-data.sqlite --ldb ~/datadir/content.ldb --mode webgraph
```

#### Error about no file features.yaml
The `run.py` program to be run from the WebGraph/code/ directory since there are some file reads with relative paths that will error if code is ran from a different directory.

----

### Requirements

#### Installation

> This project has been run and tested on *Ubuntu 18.04*

First, make sure you have `python3`, `binutils`, `pip`, `gcc `and `g++` installed. Otherwise run the following command

```bash
apt-get install binutils python3-dev python3-pip gcc g++
```

To run all tasks (Graph building, Feature extraction or Classification) on WebGraph, the crawl data used is collected using a custom version of [OpenWPM](https://github.com/sandrasiby/OpenWPM/tree/webgraph). Follow the instructions [here](https://github.com/sandrasiby/OpenWPM/tree/webgraph#readme) to setup OpenWPM in your environment. 

After OpenWPM is installed, if you haven't done it yet, activate the  *conda*  environment: 

```
conda activate openwpm
```

 go into `<project-directory>/code` in the project folder and install the python libraries in `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### Preparing Crawl Data

To generate the crawl data needed for the pipeline, you need to run a crawl using the installed OpenWPM tool. To run a crawl, first update the script `demo.py` to read in the list of sites that you want to visit. Then, run `demo.py`. 

After you run the demo, a `datadir` folder will be created in your `demo` directory. Inside the folder, you will find two database files to be used in our pipeline: `crawl-db.sqlite` and `content.ldb`

### Pipelines

The codebase consists of two pipelines: WebGraph and Robustness. We describe each of them below.

#### WebGraph Pipeline

This pipeline runs the WebGraph system, which is a graph-based Ad and Tracking Services (ATS) detection system. WebGraph takes in crawl data, builds graph representations of sites, extracts features and labels from these representations, and trains a machine learning model. 

With the WebGraph code, we present two tasks that you can run:

1. Graph Preprocessing and Feature building
2. Classification (training and testing)

#### 1. Graph preprocessing and Feature Building

In this task, WebGraph constructs the dataset for classification by:

- taking your *sqlite* and *leveldb* database files to construct a graph representation of each crawl as explained in the [paper](https://www.usenix.org/system/files/sec22summer_siby.pdf) and export it in a tabular format to a `graph.csv` file and `features.csv` file
- applying the rules from public *filterlists* to label the nodes in each graph and export it in a tabular format to a `labelled.csv` file

To run this task, run the following script:

```
python <project-directory>/code/run.py --input-db <location-to-datadir>/datadir/crawl-db.sqlite --ldb <location-to-datadir>/datadir/content.ldb --mode webgraph
```

> All additional arguments accepted by this command:
>
> - `--input-db`: the path to the `.sqlite` file generated by the crawl
> - `--ldb`: the path to the `.ldb` file generated by the crawl 
> - `--features`: the path to the `.yaml` feature categories list. A default `features.yaml` is used if unspecified.
> - `--filters`: the path to the directory to save the filter lists in. A default `filterlists` folder will be created if unspecified.
> - `--out`: the path to the directory of the output `.csv` files.
> - `--mode`: the system to run (webgraph or adgraph).

Note: With the `--mode` argument, you can also run AdGraph (we evaluate AdGraph in Section 3 of the paper).

#### 2. Classification

The classification takes in the output from Step 1 (features and labels), and performs cross validation on the data. To run this task, run the following script:

```
python <project-directory>/code/classification/classify.py --features features.csv --labels labels.csv --out results --save False --probability False --interpret False
```
> Arguments of this command:
>
> - `--features`: the path to the features.csv file
> - `--labels`: the path to the labels.csv file
> - `--out`: the path to the directory of the output files
> - `--save`: Whether to save the trained model.
> - `--probability`: Whether to save prediction probabilities.
>- `--interpret`: Whether to run interpretation on results.

<hr/>

#### Robustness Pipeline

This pipeline runs the robustness experiments performed in the paper. There are two types of robustness experiments: content and structure mutations. All the code and READMEs associated with these experiments are in the `robustness` folder.

### Data Schema

The output of the WebGraph pipeline is three files: `graph.csv`, `features.csv`, `labelled.csv`.

#### Graph

These are the columns present in the graph output under `graph.csv`

| Column               | Applies to | Description                                                  |
| -------------------- | ---------- | ------------------------------------------------------------ |
| *visit_id*           | All        | the visit id of the crawl                                    |
| *name*               | All        | the name of the node or edge                                 |
| *graph_attr*         | All        | `Node` or `Edge`                                             |
| *top_level_url*      | All        | The top level URL (page being visited)                                                   |
| *attr*               | All        | additional attributes of nodes and edges                     |
| *domain*             | All        | The parent domain of nodes or edges                          |
| *top_level_domain*   | All        | Top level domain (domain of page being visited)                                                            |
| *type*               | Node       | The type of node `Document | Element | Request | Script | Storage` |
| *document_url*       | Node       | Context of a script's execution.                                                            |
| *setter*             | Node       | The name of the node that sets a storage node.                  |
| *setting_time_stamp* | Node       | Time stamp of storage node setting.                                                           |
| *setter_domain*      | Node       | Domain of the node that sets a storage node.                                                           |
| *party*              | Node       | The partiness of a node either `first` or `third` or `N/A`   |
| *src*                | Edge       | The source node name of the edge                             |
| *dst*                | Edge       | The destination node name of the edge                        |
| *reqattr*            | Edge       | HTTP request headers                                      |
| *respattr*           | Edge       | HTTP response headers                                     |
| *response_status*    | Edge       | HTTP response status                                         |
| *content_hash*       | Edge       | Content hash if logged by OpenWPM                                                           |
| *post_body*          | Edge       | POST response body hash                                                           |
| *post_body_raw*      | Edge       | POST response body raw                                                           |

#### Features

The features in `features.csv` used are described in [features.yaml](https://github.com/spring-epfl/WebGraph/blob/main/code/features.yaml)

#### Labels

Nodes labeled by either True or False if they are blocked by filter lists or not. These are the columns present in the `labelled.csv` file.

| Column               | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| *visit_id*           | The visit id of the crawl                                    |
| *top_level_url*      | The top level URL (page being visited)                       |
| *name*               | The name of the node                                         |
| *label*              | The label of the node                                        |


<hr/>

### Code Organization

The WebGraph pipeline is in the `code` folder. The Robustness pipeline is in the `robustness` folder. 

### Paper

**WebGraph: Capturing Advertising and Tracking Information Flows for Robust Blocking**
Sandra Siby, Umar Iqbal, Steven Englehardt, Zubair Shafiq, Carmela Troncoso
_USENIX Security Symposium (USENIX), 2022_

**Abstract** -- Users rely on ad and tracker blocking tools to protect their privacy. Unfortunately, existing ad and tracker blocking tools are susceptible to mutable advertising and tracking content.  In this paper, we first demonstrate that a state-of-the-art ad and tracker blocker, AdGraph, is susceptible to such adversarial evasion techniques that are currently deployed on the web. Second, we introduce WebGraph, the first ML-based ad and tracker blocker that detects ads and trackers based on their action rather than their content. By featurizing the actions that  are fundamental to advertising and tracking information flows – e.g., storing an identifier in the browser or sharing an identifier with another tracker – WebGraph  performs nearly as well as prior approaches, but is significantly more robust to adversarial evasions. In particular, we show that WebGraph  achieves comparable accuracy to AdGraph, while significantly decreasing the success rate of an adversary from near-perfect for AdGraph to around 8% for WebGraph. Finally, we show that WebGraph remains robust to sophisticated adversaries that use adversarial evasion techniques beyond those currently deployed on the web.

The full paper can be found [here](https://www.usenix.org/system/files/sec22summer_siby.pdf).



### Citation

If you use the code/data in your research, please cite our work as follows:

```
@inproceedings{Siby22WebGraph,
  title     = {WebGraph: Capturing Advertising and Tracking Information Flows for Robust Blocking},
  author    = {Sandra Siby, Umar Iqbal, Steven Englehardt, Zubair Shafiq, Carmela Troncoso},
  booktitle = {USENIX Security Symposium (USENIX)},
  year      = {2022}
}
```

### Contact

In case of questions, please get in touch with [Sandra Siby](https://sandrasiby.github.io/). 

### Acknowledgements

Thanks to Laurent Girod and Saiid El Hajj Chehade for helping test and improve the code.


