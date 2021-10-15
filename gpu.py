from typing import List, Dict, Tuple, Set, Iterable, Any
import subprocess
from collections import defaultdict


UNK_GPU = 'gpu'
RUN_ST = 'R'
PEND_ST = 'PD'


def parse_nodes(nodes: str) -> List[str]:
  if not nodes.startswith('tir'):
    return []
  if '[' not in nodes:
    return [nodes]
  prefix, ns = nodes[:-1].split('[')
  return [prefix + n for n in ns.split(',')]


def parse_gres(gres: str) -> List[Tuple[str, int]]:
  if not gres.startswith('gpu'):
    return []
  gpus = []
  for atype in gres.split(','):
    atype = atype.split(':')
    if len(atype) == 3:
      gpus.append((atype[1], int(atype[2])))
    elif len(atype) == 2:
      gpus.append((atype[0], int(atype[1])))
    else:
      raise NotImplementedError
  return gpus


def parse_table(command: str, columns: List[str]) -> Iterable[Dict]:
  p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
  rows = p.stdout.read().decode('utf-8').rstrip('\n')
  rows = rows.split('\n')[1:]
  for row in rows:
    info: Dict[str, str] = dict(zip(columns, map(lambda x: x.strip(), row.split('\t'))))
    yield info


def pretty(gpu2status2user2count: Dict[str, Dict[str, Dict[str, int]]], gpu2count: Dict[str, int], max_display_user: int = 2, max_users_total: int = 20):
  user2count: Dict[str, int] = defaultdict(lambda: 0)
  show_user = lambda u2c, maxlen: ','.join(
    map(lambda x: f'{x[0]}:{x[1]}',
        sorted(u2c.items(), key=lambda x: (-x[1], x[0]))[:max_display_user]))[:maxlen]
  unknow_gpu = defaultdict(lambda: 0)
  unknow_gpu.update({s: sum(u2c.values()) for s, u2c in gpu2status2user2count[UNK_GPU].items()})
  del gpu2status2user2count['gpu']
  print('{:<10} | {:<25}{:>8} | {:<12}{:>8} | {:>6} | {:>6}'.format('GPU', f'top-{max_display_user}', 'Running', f'top-{max_display_user}', 'Pending', 'Free', 'Total'))
  print('-' * 88)
  allgpu: Set[str] = set(gpu2count.keys())
  inusegpu: Set[str] = set(gpu2status2user2count.keys())
  for gpu in list(inusegpu) + list(allgpu - inusegpu):
    status2user2count = gpu2status2user2count[gpu]
    status2count = defaultdict(lambda: 0)
    status2count.update({s: sum(u2c.values()) for s, u2c in status2user2count.items()})
    status2count['F'] = gpu2count[gpu] - status2count['R']
    print('{:<10} | {:<25}{:>8} | {:<12}{:>8} | {:>6} | {:>6}'.format(
      gpu,
      show_user(status2user2count[RUN_ST], 25), status2count[RUN_ST],
      show_user(status2user2count[PEND_ST], 12), status2count[PEND_ST],
      status2count['F'], gpu2count[gpu]))
    for u, c in status2user2count[RUN_ST].items():
      user2count[u] += c
  print(f"(unknown type GPU: {unknow_gpu[RUN_ST]} running {unknow_gpu[PEND_ST]} pending)")
  print('--------- top users ---------')
  for u, c in sorted(user2count.items(), key=lambda x: (-x[1], x[0]))[:max_users_total]:
    print('{:<15}\t{:10}'.format(u, c))


def get_gpu_config(filename: str = '/etc/slurm/gres.conf') -> Tuple[Dict[str, Dict[int, str]], Dict[str, int]]:
  gpu2count: Dict[str, int] = defaultdict(lambda: 0)
  node2id2gpu: Dict[str, Dict[int, str]] = defaultdict(lambda: {})
  with open(filename, 'r') as fin:
    for l in fin:
      nodes, _, gputype, gpuids = l.split()  # NodeName=tir-0-[7,9,13,15,17,19] Name=gpu Type=TITANX File=/dev/nvidia[0-3]
      nodes = parse_nodes(nodes.strip().split('=', 1)[1])
      gputype = gputype.strip().split('=', 1)[1]
      gpuid_s, gpuid_e = list(map(int, gpuids.strip().split('[', 1)[1][:-1].split('-')))  # both inclusive
      gpuid_e += 1  # exclusive
      gpu2count[gputype] += len(nodes) * (gpuid_e - gpuid_s)
      for node in nodes:
        for i in range(gpuid_s, gpuid_e):
          node2id2gpu[node][i] = gputype
  return node2id2gpu, gpu2count


def get_job_info(jobid: str) -> Dict[str, Any]:  # TODO: only work for single node jobs
  job_command = f'scontrol show jobid -dd {jobid}'
  gpu_anchor = 'GRES=gpu'
  node_anchor = ' Nodes='
  result = {'nodes': [], 'gpu_ids': []}
  p = subprocess.Popen(job_command, shell=True, stdout=subprocess.PIPE)
  job_info = p.stdout.read().decode('utf-8').rstrip('\n')
  n = job_info.find(node_anchor)
  nodes = parse_nodes(job_info[n + len(node_anchor):].split(' ', 1)[0])
  g = job_info.find(gpu_anchor)
  if g == -1:
    return result
  ids: str = job_info[g + len(gpu_anchor):].split('(', 1)[1].split(')', 1)[0][len('IDX:'):]
  if ',' in ids:  # separate ids
    _ids: List[str] = ids.split(',')
  else:
    _ids: List[str] = [ids]
  merge_ids: List[int] = []
  for __ids in _ids:
    if '-' in __ids:  # a range of ids
      s, e = __ids.split('-', 1)
      merge_ids.extend(range(int(s), int(e) + 1))
    else:
      merge_ids.append(int(__ids))
  result['nodes'] = [nodes[0]]
  result['gpu_ids'] = merge_ids
  return result


def gpu_summary():
  info_cols = ['NODELIST', 'CPUS', 'MEMORY', 'AVAIL_FEATURES', 'GRES']
  info_command = 'sinfo -o "%50N\t%10c\t%10m\t%25f\t%50G"'
  job_cols = ['JOBID', 'PARTITION', 'USER', 'ST', 'TIME', 'NODES', 'NODELIST(REASON)', 'NAME', 'TRES_PER_NODE']
  job_command = 'squeue -o "%.18i\t%.9P\t%.20u\t%.2t\t%.12M\t%.6D\t%.15R\t%.20j\t%.20b"'

  # summarize gpu
  node2id2gpu, gpu2count = get_gpu_config()

  # summarize jobs
  gpu2status2user2count: Dict[str, Dict[str, Dict[str, int]]] = \
    defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
  for job in parse_table(job_command, job_cols):
    user = job['USER']
    jobid = job['JOBID']
    gpus = parse_gres(job['TRES_PER_NODE'])
    st = job['ST']
    for gpu in gpus:
      gputype, count = gpu
      if count <= 0:
        continue
      if gputype == UNK_GPU and st == RUN_ST:
        jobinfo = get_job_info(jobid)
        ids = jobinfo['gpu_ids'][:count]
        node = jobinfo['nodes'][0]
        for i in ids:
          gpu2status2user2count[node2id2gpu[node][i]][st][user] += 1
        count -= len(ids)
      if count > 0:
        gpu2status2user2count[gputype][st][user] += count

  # display
  pretty(gpu2status2user2count, gpu2count)


gpu_summary()
