import json
import multiprocessing
import os
import re
import subprocess
import time
from multiprocessing import Process
import dateutil.parser

import numpy as np
import gpustat


class Profile:
    def __init__(self, outfile=None):
        self.outfile = outfile
        self.manager = multiprocessing.Manager()
        self.data = self.manager.list()

    @staticmethod
    def record_power(data, outfile=None):
        while True:
            q = gpustat.new_query().jsonify()
            q['query_time'] = q['query_time'].isoformat()
            data.append({
                **q,
                'power': get_power_reading(),
            })

    def __enter__(self):
        self.p = Process(target=self.record_power,
                         args=(self.data,),
                         kwargs={'outfile': self.outfile})
        self.p.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.p.terminate()
        self.p.join()
        if self.outfile is not None:
            with open(self.outfile, 'w') as f:
                json.dump(list(self.data), f)
            print(f'Wrote output to {self.outfile}')


class ProfileRun:

    def __init__(self, data):
        self.data = data

    @classmethod
    def from_json(cls, filename):
        with open(filename) as f:
            data = json.load(f)
        return cls(data)

    def query_time(self, index):
        return dateutil.parser.parse(self.data[index]['query_time'])

    def power(self, index):
        return int(self.data[index]['power'])

    def gpu_powers(self, index):
        return [d['power.draw'] for d in self.data[index]['gpus']]

    def gpu_temps(self, index):
        return [d['temperature.gpu'] for d in self.data[index]['gpus']]

    def gpu_temp(self, index, gpu_index=None):
        if gpu_index is None:
            return np.mean(self.gpu_temps(index))
        else:
            return self.gpu_temps(index)[gpu_index]

    def gpu_utils(self, index):
        return [d['utilization.gpu'] for d in self.data[index]['gpus']]

    def gpu_util(self, index, gpu_index=None):
        if gpu_index is None:
            return np.mean(self.gpu_utils(index))
        else:
            return self.gpu_utils(index)[gpu_index]

    def gpu_mems(self, index):
        return [d['memory.used'] for d in self.data[index]['gpus']]

    def gpu_mem(self, index, gpu_index=None):
        if gpu_index is None:
            return np.mean(self.gpu_mems(index))
        else:
            return self.gpu_mems(index)[gpu_index]

    def gpu_power(self, index, gpu_index=None):
        if gpu_index is None:
            return sum(self.gpu_powers(index))
        else:
            return self.gpu_powers(index)[gpu_index]

    def total_power(self):
        return sum(self.power(i) for i in range(len(self))) / self.total_time()

    def total_gpu_power(self, gpu_index=None):
        return np.sum(self.gpu_power_profile(gpu_index=gpu_index)) / self.total_time()

    def average_power(self):
        return np.mean([self.power(i) for i in range(len(self))])

    def time_profile(self):
        return [(self.query_time(i) - self.query_time(0)).total_seconds() for i in range(len(self))]

    def total_time(self):
        return self.time_profile()[-1]

    def power_profile(self):
        return [self.power(i) for i in range(len(self))]

    def gpu_power_profile(self, gpu_index=None):
        return [self.gpu_power(i, gpu_index) for i in range(len(self))]

    def gpu_util_profile(self, gpu_index=None):
        return [self.gpu_util(i, gpu_index=gpu_index) for i in range(len(self))]

    def gpu_mem_profile(self, gpu_index=None):
        return [self.gpu_mem(i, gpu_index=gpu_index) for i in range(len(self))]

    def gpu_temp_profile(self, gpu_index=None):
        return [self.gpu_temp(i, gpu_index=gpu_index) for i in range(len(self))]

    def __len__(self):
        return len(self.data)

    def print_stats(self, gpu_index=None):
        print(f'Total time:      {self.total_time()}')
        print(f'Total Power:     {self.total_power() / (60 * 60)} Wh')
        print(f'Total GPU Power: {self.total_gpu_power(gpu_index=gpu_index) /(60 * 60)} Wh')


def get_power_reading():
    out = subprocess.Popen(['sudo', '/home/software/perftools/0.1/bin/satori-ipmitool'],
                           stdout=subprocess.PIPE)
    out = subprocess.Popen(['grep', 'Instantaneous power reading'], stdin=out.stdout, stdout=subprocess.PIPE)
    out = subprocess.run(['awk', '{print $4}'], stdin=out.stdout, stdout=subprocess.PIPE)
    if out.returncode == 0:
        return out.stdout.decode().strip()


class Profile:
    def __init__(self, outfile=None):
        self.outfile = outfile
        self.manager = multiprocessing.Manager()
        self.data = self.manager.list()

    @staticmethod
    def record_power(data, outfile=None):
        while True:
            q = gpustat.new_query().jsonify()
            q['query_time'] = q['query_time'].isoformat()
            data.append({
                **q,
                'power': get_power_reading(),
            })

    def __enter__(self):
        self.p = Process(target=self.record_power,
                         args=(self.data,),
                         kwargs={'outfile': self.outfile})
        self.p.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.p.terminate()
        self.p.join()
        if self.outfile is not None:
            with open(self.outfile, 'w') as f:
                json.dump(list(self.data), f)
            print(f'Wrote output to {self.outfile}')


class ProfileRun:

    def __init__(self, data):
        self.data = data

    @classmethod
    def from_json(cls, filename):
        with open(filename) as f:
            data = json.load(f)
        return cls(data)

    def query_time(self, index):
        return dateutil.parser.parse(self.data[index]['query_time'])

    def power(self, index):
        return int(self.data[index]['power'])

    def _gpus(self, index):
        return self.data[index]['gpus']

    def _gpus_stats(self, index, name):
        return [d[name] for d in self._gpus(index)]

    def gpu_powers(self, index):
        return self._gpus_stats(index, 'power.draw')

    def gpu_temps(self, index):
        return self._gpus_stats(index, 'temperature.gpu')

    def gpu_utils(self, index):
        return self._gpus_stats(index, 'utilization.gpu')

    def gpu_mems(self, index):
        return self._gpus_stats(index, 'memory.used')

    def gpu_temp(self, index, gpu_index=None):
        out = self.gpu_temps(index)
        return np.mean(out) if gpu_index is None else out[gpu_index]

    def gpu_util(self, index, gpu_index=None):
        out = self.gpu_utils(index)
        return np.mean(out) if gpu_index is None else out[gpu_index]

    def gpu_mem(self, index, gpu_index=None):
        out = self.gpu_mems(index)
        return np.mean(out) if gpu_index is None else out[gpu_index]

    def gpu_power(self, index, gpu_index=None):
        out = self.gpu_powers(index)
        return np.mean(out) if gpu_index is None else out[gpu_index]

    def total_power(self):
        return sum([self.power(i) for i in range(len(self))]) / self.total_time()

    def total_gpu_power(self, gpu_index=None):
        return np.sum(self.gpu_power_profile(gpu_index=gpu_index)) / self.total_time()

    def average_power(self):
        return np.mean([self.power(i) for i in range(len(self))])

    def time_profile(self):
        return [(self.query_time(i) - self.query_time(0)).total_seconds() for i in range(len(self))]

    def total_time(self):
        return self.time_profile()[-1]

    def _profile(self, name):
        f = getattr(self, name)
        return [f(i) for i in range(len(self))]

    def _gprofile(self, name, gpu_index):
        f = getattr(self, name)
        return [f(i, gpu_index=gpu_index) for i in range(len(self))]

    def power_profile(self):
        return self._profile('power')

    def gpu_power_profile(self, gpu_index=None):
        return self._gprofile('gpu_power', gpu_index)

    def gpu_util_profile(self, gpu_index=None):
        return self._gprofile('gpu_util', gpu_index)

    def gpu_mem_profile(self, gpu_index=None):
        return self._gprofile('gpu_mem', gpu_index)

    def gpu_temp_profile(self, gpu_index=None):
        return self._gprofile('gpu_temp', gpu_index)

    def __len__(self):
        return len(self.data)

    def print_stats(self, gpu_index=None):
        print(f'Total time:      {self.total_time()}')
        print(f'Total Power:     {self.total_power() / (60 * 60)} Wh')
        print(f'Total GPU Power: {self.total_gpu_power(gpu_index=gpu_index) /(60 * 60)} Wh')


def get_satori_power_reading():
    out = subprocess.Popen(['sudo', '/home/software/perftools/0.1/bin/satori-ipmitool'],
                           stdout=subprocess.PIPE)
    out = subprocess.Popen(['grep', 'Instantaneous power reading'], stdin=out.stdout, stdout=subprocess.PIPE)
    out = subprocess.run(['awk', '{print $4}'], stdin=out.stdout, stdout=subprocess.PIPE)
    # root = os.path.dirname(os.path.abspath(__file__))
    # out = subprocess.run(['sh', f'{root}/get_power'], stdout=subprocess.PIPE)
    if out.returncode == 0:
        return out.stdout.decode().strip()


BASE = "/sys/class/powercap/"
DELAY = .1  # in seconds
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class RAPLFile:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.baseline = []
        self.process = []
        self.recent = 0
        self.num_process_checks = 0
        self.process_average = 0
        self.baseline_average = 0

    def set_recent(self, val):
        self.recent = val

    def create_gpu(self, baseline_average, process_average):
        self.baseline_average = baseline_average
        self.process_average = process_average

    def average(self, baseline_checks):
        self.process_average = sum(self.process) / self.num_process_checks
        self.baseline_average = sum(self.baseline) / baseline_checks

    def __repr__(self):
        return f'{self.name} {self.path} {self.recent}'

    def __str__(self):
        return f'{self.name} {self.path} {self.recent}'


def reformat(name, multiple_cpus):
    """ Renames the RAPL files for better readability/understanding """
    if 'package' in name:
        if multiple_cpus:
            name = "CPU" + name[-1]  # renaming it to CPU-x
        else:
            name = "Package"
    if name == 'core':
        name = "CPU"
    elif name == 'uncore':
        name = "GPU"
    elif name == 'dram':
        name = name.upper()
    return name


def get_files():
    """ Gets all the RAPL files with their names on the machine
        Returns:
            filenames (list): list of RAPLFiles
    """
    # Removing the intel-rapl folder that has no info
    files = list(filter(lambda x: ':' in x, os.listdir(BASE)))
    names = {}
    cpu_count = 0
    multiple_cpus = False
    for file in files:
        if (re.fullmatch("intel-rapl:.", file)):
            cpu_count += 1

    if cpu_count > 1:
        multiple_cpus = True

    for file in files:
        path = BASE + '/' + file + '/name'
        with open(path) as f:
            name = f.read()[:-1]
            renamed = reformat(name, multiple_cpus)
        names[renamed] = BASE + file + '/energy_uj'

    filenames = []
    for name, path in names.items():
        name = RAPLFile(name, path)
        filenames.append(name)

    return filenames, multiple_cpus


def to_joules(ujoules):
    """ Converts from microjoules to joules """
    return ujoules * 10**(-6)


def measure_files(files, delay=1):
    """ Measures the energy output of all packages which should give total power usage
    Parameters:
        files (list): list of RAPLFiles
        delay (int): RAPL file reading rate in ms
    Returns:
        files (list): list of RAPLfiles with updated measurements
    """

    files = list(map(start, files))
    time.sleep(delay)
    files = list(map(lambda x: end(x, delay), files))  # need lambda to pass in delay
    return files


def read(file):
    """ Opens file and reads energy measurement """
    if file == "":
        return 0
    with open(file, 'r') as f:
        return to_joules(int(f.read()))


def start(raplfile):
    measurement = read(raplfile.path)
    raplfile.recent = measurement
    return raplfile


def end(raplfile, delay):
    measurement = read(raplfile.path)
    raplfile.recent = (measurement - raplfile.recent) / delay
    return raplfile


def get_total(raplfiles, multiple_cpus):
    total = 0
    if multiple_cpus:
        for file in raplfiles:
            if "CPU" in file.name:
                total += file.recent
    else:
        for file in raplfiles:
            if file.name == "Package":
                total = file.recent
    if (total):
        return total
    return 0


def cpu_watts():
    files, multiple_cpus = get_files()
    files = measure_files(files, DELAY)
    reading = get_total(files, multiple_cpus)
    return reading


PLATFORM = subprocess.run(['uname', '-i'], stdout=subprocess.PIPE).stdout.decode().strip()
if PLATFORM == 'ppc64le':
    get_power_reading = get_satori_power_reading
else:
    get_power_reading = cpu_watts


def timeit(func, num_runs=20):
    times = []
    for i in range(num_runs):
        ts = time.time()
        result = func()
        te = time.time()
        times.append((te - ts) * 1000)
    print(f'{func.__name__} {np.mean(times):2.2f} ms')


if __name__ == '__main__':

    # timeit(get_satori_power_reading)
    # timeit(old_get_satori_power_reading)

    times = []
    num_runs = 50
    for i in range(num_runs):
        ts = time.time()
        gpustat.new_query().jsonify()
        te = time.time()
        times.append((te - ts) * 1000)
    print(f'gpustat {np.mean(times):2.2f} ms')
