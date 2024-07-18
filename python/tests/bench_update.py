import pathlib
import sys
import enum
import time
import asyncio
import threading

import deglib
from deglib.utils import get_current_rss_mb, StopWatch


class DataStreamType(enum.Enum):
    AddAll = enum.auto()
    AddHalf = enum.auto()
    AddAllRemoveHalf = enum.auto()
    AddHalfRemoveAndAddOneAtATime = enum.auto()


class DatasetName(enum.Enum):
    Audio = enum.auto()
    Enron = enum.auto()
    Sift1m = enum.auto()
    Glove = enum.auto()


def main():
    if deglib.avx_usable():
        print("use AVX2  ...")
    elif deglib.sse_usable():
        print("use SSE  ...")
    else:
        print("use arch  ...")

    print("Actual memory usage: {} Mb".format(get_current_rss_mb()))

    data_path: pathlib.Path = pathlib.Path("/home/ubuntu/data/new_filtered_ann_datasets/")
    gt_path: pathlib.Path = pathlib.Path("/home/ubuntu/data/ann_rolling_update_gt/redcaps_cosine_5000_100/")

    dataset_name = "redcaps"
    dataset_full_name = "redcaps-512-angular"
    repository_file = data_path / dataset_name / (dataset_full_name+".hdf5")
    metric = "cosine"
    gt_suffix = "out.hdf5"
    rolling_update_gt_file = gt_path / gt_suffix
    test_graph_fresh_update(repository_file, rolling_update_gt_file, d=20, k_ext=40, eps_ext=0.3, k_opt=20, eps_opt=0.001, i_opt=5)


def test_graph_fresh_update(
        repository_file: pathlib.Path, rolling_update_gt_file: pathlib.Path, d: int, k_ext: int,
        eps_ext: float, k_opt: int, eps_opt: float, i_opt: int
):
    rnd = deglib.Mt19937()  # default 7
    metric = deglib.Metric.Cosine  # default metric
    swap_tries = 0  # additional swap tries between the next graph extension
    additional_swap_tries = 0  # increase swap try count for each successful swap
    # load data
    print("Load Data")
    repository, query_repository = deglib.repository.parse_ann_benchmark(repository_file) # repository is a float np
    repository = repository[:10000]
    # TODO: report actual mem usage
    print("Actual memory usage: {} Mb after loading data".format(get_current_rss_mb()))

    # create a new graph
    print("Setup empty graph with {} vertices in {}D feature space".format(repository.shape[0], repository.shape[1]))
    dims = repository.shape[1]
    max_vertex_count = repository.shape[0]
    feature_space = deglib.FloatSpace(dims, metric)
    graph = deglib.graph.SizeBoundedGraph(max_vertex_count, d, feature_space)
    # TODO: report actual mem usage
    print("Actual memory usage: {} Mb after setup empty graph".format(get_current_rss_mb()))

    # create a graph builder to add vertices to the new graph and improve its edges
    print("Start graph builder")
    builder = deglib.builder.EvenRegularGraphBuilder(
        graph, rnd, k_ext, eps_ext, k_opt, eps_opt, i_opt, swap_tries, additional_swap_tries
    )

    # provide all features to the graph builder at once. In an online system this will be called multiple times
    base_size = repository.shape[0]
    base_size //= 2

    def add_entry(label):
        feature = repository[label]
        # feature_vector = std::vector<std::byte>{feature, feature + dims * sizeof(float)};
        builder.add_entry(label, feature)

    for i in range(base_size):
        add_entry(i)

    print("Actual memory usage: {} Mb after setup graph builder (including input)".format(get_current_rss_mb()))
    

    # check the integrity of the graph during the graph build process
    log_after = 100000

    print("Start building")
    start = time.perf_counter()
    duration = 0

    def improvement_callback(status):
        size = graph.size()
        nonlocal duration
        nonlocal start

        if status.step % log_after == 0 or size == base_size:
            duration += time.perf_counter() - start
            avg_edge_weight = deglib.analysis.calc_avg_edge_weight(graph, 1)
            weight_histogram_sorted = deglib.analysis.calc_edge_weight_histogram(graph, True, 1)
            weight_histogram = deglib.analysis.calc_edge_weight_histogram(graph, False, 1)
            valid_weights = deglib.analysis.check_graph_weights(graph) and deglib.analysis.check_graph_regularity(graph, size, True)
            connected = deglib.analysis.check_graph_connectivity(graph)

            print("{:7} vertices, {:6.3f}s, {:4} / {:4} improv, Q: {:.3f} -> Sorted:{}, InOrder:{}, {} connected & {}, RSS {}".format(
                    size, duration, status.improved, status.tries, avg_edge_weight,
                    " ".join(str(h) for h in weight_histogram_sorted), " ".join(str(h) for h in weight_histogram),
                    "" if connected else "not", "valid" if valid_weights else "invalid",
                    get_current_rss_mb()
                )
            )
            start = time.perf_counter()
        elif status.step % (log_after//10) == 0:
            duration += time.perf_counter() - start
            avg_edge_weight = deglib.analysis.calc_avg_edge_weight(graph, 1)
            connected = deglib.analysis.check_graph_connectivity(graph)

            print("{:7} vertices, {:6.3f}s, {:4} / {:4} improv, AEW: {:.3f}, {} connected, RSS {}".format(
                size, duration, status.improved, status.tries, avg_edge_weight,
                "" if connected else "not", get_current_rss_mb())
            )
            start = time.perf_counter()

    # start the build process
    stopwatch = StopWatch()
    # builder.build(improvement_callback, False)
    async def build_infinite():
        await builder.build(None, True)
    def build_wrapper():
        asyncio.run(build_infinite())
    _thread = threading.Thread(target=build_wrapper, args=())
    _thread.start()
    print("Starting to test sliding window update")
    rolling_update_gt = deglib.repository.get_rolling_update_gt(rolling_update_gt_file)
    def test_rolling_update(batch_id):
        deglib.benchmark.test_graph_anns(graph, query_repository, rolling_update_gt[batch_id], repeat=1, k=10)
    batch_size = base_size // 100
    test_rolling_update(0)
    for batch_id in range(100):
        print("Batch ", batch_id)
        duration = stopwatch.get_elapsed_time_micro() / 1000000
        print("Actual memory usage: {} Mb after building the graph in {:.4} secs".format(
            get_current_rss_mb(), float(duration)
        ))
        for j in range(batch_size):
            builder.remove_entry(batch_id * 100 + j)
            add_entry(batch_id * 100 + j + base_size)
        test_rolling_update(batch_id + 1)
    builder.stop()

    # store the graph
    # graph.save_graph(graph_file)
    # print("The graph contains {} non-RNG edges".format(deglib.analysis.calc_non_rng_edges(graph)))


def test_graph(query_file: pathlib.Path, gt_file: pathlib.Path, graph_file: pathlib.Path, repeat: int, k: int):
    # load an existing graph
    print("Load graph {}".format(graph_file))
    graph = deglib.graph.load_readonly_graph(graph_file)
    print("Actual memory usage: {} Mb after loading the graph".format(get_current_rss_mb()))

    query_repository = deglib.repository.fvecs_read(query_file)
    print("{} Query Features with {} dimensions".format(query_repository.shape[0], query_repository.shape[1]))

    ground_truth = deglib.repository.ivecs_read(gt_file)
    print("{} ground truth {} dimensions".format(ground_truth.shape[0], ground_truth.shape[1]))

    deglib.benchmark.test_graph_anns(graph, query_repository, ground_truth, repeat, k)


if __name__ == '__main__':
    main()
