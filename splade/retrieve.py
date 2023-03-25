import hydra
from omegaconf import DictConfig

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .evaluate import evaluate
from .models.models_utils import get_model
from .tasks.transformer_evaluator import SparseRetrieval
from .utils.utils import get_dataset_name, get_initialize_config
import h5py
import numpy as np
import pandas as pd
import time
from scipy.sparse import csr_matrix

FILTER_BY_TOPIC = True

FILTER_TOPICS_BY_VALUE_THRESHOLD = False
PASSAGE_TOPIC_THRESHOLD = 0.75
QUERY_TOPIC_THRESHOLD = 0.1

FILTER_TOPICS_BY_RANKING = True
PASSAGE_TOPIC_RANK_THRESHOLD = 4
QUERY_TOPIC_RANK_THRESHOLD = 2

def topic_set_to_int(topic_set):
    bins = ['0' for i in range(19)]
    for topic_i in topic_set:
        bins[topic_i] = '1'

    bin_num = '0b' + ''.join(bins)
    return int(bin_num, 2)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def retrieve_evaluate(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)

    model = get_model(config, init_dict)

    passage_topic_scores = []
    query_topic_scores = []
    if FILTER_BY_TOPIC:
        # Construct sparse map of passage scores
        topic_index_file = h5py.File('data/toy_data/full_collection/full_collection_classifications.h5', 'r')
        topic_index = topic_index_file['classifications'][()]
        topic_index_file.close()

        for passage_i in range(topic_index.shape[0]):
            topic_set = set()

            if FILTER_TOPICS_BY_VALUE_THRESHOLD:
                for topic_i in range(topic_index.shape[1]):
                    topic_score = topic_index[passage_i, topic_i]
                    if topic_score > PASSAGE_TOPIC_THRESHOLD:
                        topic_set.add(topic_i)
            elif FILTER_TOPICS_BY_RANKING:
                top_topics = np.argsort(topic_index[passage_i, :])[:PASSAGE_TOPIC_RANK_THRESHOLD]
                topic_set = set(top_topics)

            passage_topic_scores.append(topic_set_to_int(topic_set))

        # Get the topic of the query
        q_topic_index_file = h5py.File('data/toy_data/dev_queries/query_classifications.h5', 'r')
        q_topic_index = q_topic_index_file['classifications'][()]
        q_topic_index_file.close()

        for query_i in range(q_topic_index.shape[0]):
            topic_set = set()

            if FILTER_TOPICS_BY_VALUE_THRESHOLD:
                for topic_i in range(q_topic_index.shape[1]):
                    topic_score = q_topic_index[query_i, topic_i]
                    if topic_score > QUERY_TOPIC_THRESHOLD:
                        topic_set.add(topic_i)

            elif FILTER_TOPICS_BY_RANKING:
                top_topics = np.argsort(q_topic_index[query_i, :])[:QUERY_TOPIC_RANK_THRESHOLD]
                topic_set = set(top_topics)

            query_topic_scores.append(topic_set_to_int(topic_set))

    batch_size = 1
    # NOTE: batch_size is set to 1, currently no batched implem for retrieval (TODO)
    for data_dir in set(exp_dict["data"]["Q_COLLECTION_PATH"]):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                        max_length=model_training_config["max_length"], batch_size=batch_size,
                                        shuffle=False, num_workers=1)
        evaluator = SparseRetrieval(config=config, model=model, dataset_name=get_dataset_name(data_dir),
                                    compute_stats=True, dim_voc=model.output_dim)

        start = time.time()
        evaluator.retrieve(q_loader,
                           top_k=exp_dict["config"]["top_k"],
                           threshold=exp_dict["config"]["threshold"],
                           passage_topic_scores=passage_topic_scores,
                           query_topic_scores=query_topic_scores,
                           filter_by_topic=FILTER_BY_TOPIC)
        finish = time.time()

        print(f"Retrieval took {finish-start} seconds")

    evaluate(exp_dict)



if __name__ == "__main__":
    retrieve_evaluate()
