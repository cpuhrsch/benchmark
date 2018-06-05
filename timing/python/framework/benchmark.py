import logging
import numpy as np
import csv
import random
import utils as bench_utils


class AttrDict(dict):
    def __repr__(self):
        return ", ".join(k + "=" + str(v) for k, v in self.items())

    def __getattr__(self, name):
        return self[name]


class BenchmarkResult(dict):
    """
    Initialize with a double of seconds
    """

    def __init__(self, time, cpu, num_iter):
        self.time = float(time)
        self.cpu = float(cpu)
        self.num_iter = num_iter

    def get_time(self, unit="us"):
        return self.time * 1e6

    def get_cpu(self, unit="us"):
        return self.cpu * 1e6

    def get_num_iter(self):
        """
        Number of iterations run to get timings
        """
        return self.num_iter

    def __repr__(self):
        return ", ".join(k + "=" + str(v) for k, v in self.items())


class BenchmarkResults(object):
    def __init__(self):
        self.results = []

    def time_mean(self, unit="us"):
        return int(
            np.mean(list(map(lambda x: x.get_time(unit), self.results)))
        )

    def time_std(self, unit="us"):
        return int(np.std(list(map(lambda x: x.get_time(unit), self.results))))

    def cpu_mean(self, unit="us"):
        return int(np.mean(list(map(lambda x: x.get_cpu(unit), self.results))))

    def cpu_std(self, unit="us"):
        return int(np.std(list(map(lambda x: x.get_cpu(unit), self.results))))

    def iter_mean(self):
        return int(
            np.mean(list(map(lambda x: x.get_num_iter(), self.results)))
        )

    def append(self, result):
        assert isinstance(result, BenchmarkResult)
        self.results.append(result)


class Benchmark(object):
    def __init__(self):
        if "args" not in dir(self):
            logging.warn(
                "Benchmark " + str(self.__class__) + " has no args set."
                " Using empty dictionary."
            )
            self.args = [{}]
        cpus = bench_utils.get_cpu_list()
        for cpu in cpus:
            bench_utils.check_cpu_governor(cpu)
        self.state = {}


class ListBenchmark(Benchmark):
    """
    Basic benchmark class. Expects list of arguments and processes one at a
    time.
    """

    def __init__(self):
        super(ListBenchmark, self).__init__()
        if not isinstance(self.args, list):
            raise TypeError("args needs to be a list of arguments")


class GridBenchmark(Benchmark):
    """
    Creates a grid of arguments and calls benchmark with each.
    The arguments must be primited types such as strings or
    numbers and will be shallow(!) copied as part of the setup
    """

    def __init__(self):
        super(GridBenchmark, self).__init__()
        if not isinstance(self.args, dict):
            raise TypeError("args needs to be a dict of arguments")
        self.args = bench_utils.grid(self.args)


# TODO
# benchmark [--benchmark_list_tests={true|false}]
#           [--benchmark_filter=<regex>]
#           [--benchmark_format=<console|json|csv>]
#           [--benchmark_out=<filename>]
#           [--benchmark_out_format=<json|console|csv>]
#           [--benchmark_color={auto|true|false}]
#           [--benchmark_counters_tabular={true|false}]
#           [--v=<verbosity>]
# DONE
#           [--benchmark_min_time=<min_time>]
#           [--benchmark_repetitions=<num_repetitions>]
#           [--benchmark_report_aggregates_only={true|false}

# TODO: Write general setup script to check for environment setup
# TODO: Add functionality for user counters e.g. custom timings
# TODO: Allow option to output csv directly instead of pretty print
# - better for remote


def run_func_benchmark(func, arg, state, settings):
    num_iter = 0
    start = bench_utils.timer()
    end = start
    cpu_start = bench_utils.cpu_timer()
    while (
        end - start <= settings.benchmark_min_time
        or num_iter <= settings.benchmark_min_iter
    ):
        if (
            num_iter >= settings.benchmark_max_iter
            or end - start >= settings.benchmark_max_time
        ):
            break
        func(state, AttrDict(arg))
        end = bench_utils.timer()
        num_iter += 1
    result = BenchmarkResult(
        end - start, bench_utils.cpu_timer() - cpu_start, num_iter
    )
    return result


def make_print_row(row, row_format, header):
    status_str = ""
    for i in range(len(header)):
        v = row[header[i]]
        status_str += row_format[header[i]].format(str(v))
    return status_str


def make_pretty_print_row_format(args, header, header_labels, header_init):
    max_name_lens = {}
    for i in range(len(header)):
        max_name_lens[header[i]] = header_init[i]
    for arg in args:
        for k, v in arg.items():
            if k not in max_name_lens:
                max_name_lens[k] = len(str(k)) + 3
                header += [k]
                header_labels += [k]

            max_name_lens[k] = max(max_name_lens[k], len(str(v)) + 3)
    row_format = {}
    for i in range(len(header)):
        row_format[header[i]] = "{:>" + str(max_name_lens[header[i]]) + "}"
    return row_format


def append_row(rows, row):
    for k, v in row.items():
        rows[k].append(v)


def create_jobs(obj):
    jobs = []
    for func in dir(obj):
        if func.startswith("benchmark"):
            for arg in obj.args:
                config = AttrDict()
                config.func = func
                config.arg = arg.copy()
                jobs.append(config)
    random.shuffle(jobs)
    return jobs


def run_benchmark_job(job, obj, settings):
    arg = job.arg
    func = getattr(obj, job.func)
    row = arg.copy()
    row["repetitions"] = settings.benchmark_repetitions
    results = BenchmarkResults()
    for _ in range(settings.benchmark_repetitions):
        if "setup" in dir(obj):
            obj.setup(obj.state, AttrDict(arg))
        results.append(run_func_benchmark(func, arg, obj.state, settings))
        if "teardown" in dir(obj):
            obj.teardown(obj.state, AttrDict(arg))
    row["time_mean"] = results.time_mean()
    row["time_std"] = results.time_std()
    row["cpu_mean"] = results.cpu_mean()
    row["cpu_std"] = results.cpu_std()
    row["iter_mean"] = results.iter_mean()
    return row


def calculate_progress(job_number, max_job_number, time_elapsed, info_format):
    avg_time = (float(time_elapsed)) / (job_number)
    time_left = int((max_job_number - job_number) * avg_time)
    info = info_format.format(
        "{}/{}".format(job_number, max_job_number),
        int(time_left / 60),
        "{:>02d}".format(time_left % 60),
    )
    return info


def run_benchmark(obj, name, settings):
    """
    Create benchmark table. All times are in microseconds.
    """
    header = [
        "benchmark",
        "time_mean",
        "time_std",
        "cpu_mean",
        "cpu_std",
        "iter_mean",
        "repetitions",
    ]
    header_label = [
        "Benchmark",
        "Time mean (us)",
        "Time std (us)",
        "CPU mean (us)",
        "CPU std (us)",
        "Iter. mean",
        "Rep.",
    ]
    header_init = [max(12, len(name) + 3), 18, 18, 18, 18, 13, 14]

    row_format = make_pretty_print_row_format(
        obj.args, header, header_label, header_init
    )
    rows = {}
    for head in header:
        rows[head] = []

    jobs = create_jobs(obj)
    info_format = "{:>15} {:>10}:{:>2}"
    hstr = info_format.format("Job number", "ETA (hh", "mm)")
    for i in range(len(header)):
        hstr += row_format[header[i]].format(str(header_label[i]))
    print(len(hstr) * "-")
    print(hstr)
    print(len(hstr) * "-")
    out_csv_obj = None
    if settings.benchmark_out:
        out_csv_fd = open(settings.benchmark_out, "w")
        out_csv_obj = csv.DictWriter(out_csv_fd, header)
        out_csv_obj.writeheader()
    total_time = bench_utils.timer()
    for i in range(len(jobs)):
        row = run_benchmark_job(jobs[i], obj, settings)
        row["benchmark"] = name
        append_row(rows, row)
        if out_csv_obj:
            out_csv_obj.writerow(row)
            out_csv_fd.flush()
        info = calculate_progress(
            i + 1, len(jobs), bench_utils.timer() - total_time, info_format
        )
        print(info + make_print_row(row, row_format, header))


def create_benchmark_object(benchmark_class):
    try:
        assert issubclass(benchmark_class, Benchmark)
    except TypeError as e:
        raise TypeError(
            str(benchmark_class) + " must be subclass of Benchmark"
        )
    return benchmark_class()
