from typing import List, Set

import numpy as np
import tqdm

import deglib.graph
import deglib.repository
import deglib.utils


def test_graph_anns(
        graph: deglib.graph.ReadOnlyGraph, query_repository: deglib.repository.StaticFeatureRepository,
        ground_truth: np.ndarray, ground_truth_dims: int, repeat: int, k: int
):
    # entry_vertex_id = get_near_avg_entry_index(graph)

    entry_vertex_indices = graph.get_entry_vertex_indices()
    print("internal id {}".format(graph.get_internal_index(entry_vertex_indices[0])))

    # test ground truth
    print("Parsing gt:")
    answer = get_ground_truth(ground_truth, query_repository.size(), ground_truth_dims, k)
    print("Loaded gt:")

    # try different eps values for the search radius
    # eps_parameter = [0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.15, 0.2]                   # crawl
    # eps_parameter = [0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.18, 0.2]                    # enron
    # eps_parameter = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]                                  # UQ-V
    # eps_parameter = [0.00, 0.03, 0.05, 0.07, 0.09, 0.12, 0.2, 0.3]                    # audio
    eps_parameter = [0.01, 0.05, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]                    # SIFT1M k=100
    # eps_parameter = [0.01, 0.05, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]                    # SIFT1M k=100
    # eps_parameter = [100, 140, 171, 206, 249, 500, 1000]                              # greeedy search SIFT1M k=100
    # eps_parameter = [0.00, 0.01, 0.05, 0.1, 0.15, 0.2]                                # SIFT1M k=1
    # eps_parameter = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2]  # clipfv
    # eps_parameter = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08 0.09, 0.1, 0.2]   # gpret
    # eps_parameter = [0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4]                           # GloVe
    # eps_parameter = [0.01, 0.06, 0.07, 0.08, 0.09, 0.11, 0.13, 0.15, 0.20]            # GloVe DEG90

    for eps in eps_parameter:
        stopwatch = deglib.utils.StopWatch()
        recall = 0.0
        for i in range(repeat):
            recall = test_approx_anns(graph, entry_vertex_indices, query_repository, answer, eps, k)
        time_us_per_query = (stopwatch.get_elapsed_time_micro() / query_repository.size()) / repeat

        print("eps {:.2f} \t recall {:.5f} \t time_us_per_query {:6}us".format(eps, recall, time_us_per_query))
        if recall > 1.0:
            break


def get_near_avg_entry_index(graph: deglib.graph.ReadOnlyGraph, verbose: bool = False) -> int:
    """
    Get the internal index of the graph, that is closest to the average vector in the graph.
    """
    feature_dims = graph.get_feature_space().dim()
    graph_size = graph.size()
    avg_fv = np.zeros(feature_dims, dtype=np.float32)
    progress_func = tqdm.tqdm if verbose else deglib.utils.no_tqdm
    for i in progress_func(range(graph_size), desc='calculating average vector'):
        fv = graph.get_feature_vector(i)
        avg_fv += fv

    avg_fv /= graph_size
    seed = [graph.get_internal_index(0)]
    result_queue = graph.search(seed, avg_fv, 0.1, 30)
    entry_vertex_id = result_queue.top().get_internal_index()

    return entry_vertex_id


def get_ground_truth(ground_truth: np.ndarray, ground_truth_size: int, ground_truth_dims: int, k: int) -> List[Set[int]]:
    if ground_truth_dims < k:
        raise ValueError("Ground truth data has only {} elements but need {}".format(ground_truth_dims, k))

    answers = [set() for i in range(ground_truth_size)]
    for i in range(ground_truth_size):
        gt = answers[i]
        for j in range(k):
            gt.add(ground_truth[i, j])

    return answers


def test_approx_anns(
        graph: deglib.graph.ReadOnlyGraph, entry_vertex_indices: List[int],
        query_repository: deglib.repository.StaticFeatureRepository, ground_truth: List[Set[int]], eps: float, k: int
):
    total = 0
    correct = 0
    for i in range(query_repository.size()):
        query = query_repository.get_feature(i)
        result_queue = graph.search(entry_vertex_indices, query, eps, k)
        # result_queue = graph.search(entry_vertex_indices, query, eps, k, graph.size())  # max distance calcs

        if result_queue.size() != k:
            raise ValueError("ANNS with k={} got only {} results for query {}".format(k, result_queue.size(), i))

        total += result_queue.size()
        gt = ground_truth[i]
        # checked_ids = set()  # additional check
        while not result_queue.empty():
            internal_index = result_queue.top().get_internal_index()
            external_id = graph.get_external_label(internal_index)
            if external_id in gt:
                correct += 1
            result_queue.pop()
            # checked_ids.add(internal_index)

        # if len(checked_ids) != k:
        #     raise ValueError("ANNS with k={} got only {} unique ids".format(k, len(checked_ids)))

    return 1.0 * correct / total