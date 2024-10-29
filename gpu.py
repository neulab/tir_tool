#!/usr/bin/env python3

import sys
from typing import List, Dict, Tuple, Set, Iterable, Any
import argparse
import subprocess
from collections import defaultdict
import re


UNK_GPU = 'gres/gpu'
RUN_ST = 'R'
PEND_ST = 'PD'
FLAG = {'verbose': False}
# The reasons which we want to count as "pending".  E.g., don't count jobs
# pending because they exceed the partition time limit, are waiting on
# a dependency, or can't get a requested node because it is down.
PENDING_REASONS = (
  "(Resources)",
  "(Priority)",
)



def parse_nodes(nodes: str) -> List[str]:
  # Example: babel-1-[23,27,31],babel-2-12
  nodelist = []
  prefix = ''
  for s in nodes.split(','):
    if '[' in s:
      prefix, ns = s.split('[')
      nodelist.append(prefix + ns)
    elif ']' in s:
      nodelist.append(prefix + s[:-1])
      prefix = ''
    else:
      nodelist.append(prefix + s)
  return nodelist


def parse_gres(gres: str) -> List[Tuple[str, int]]:
  if not gres.startswith('gres/gpu'):
    return []
  gpus = []
  for atype in gres.split(','):
    atype = atype.split(':')
    if len(atype) == 3:
      gpus.append((atype[1], int(atype[2])))
    elif len(atype) == 2:
      try:
        gpus.append((atype[0], int(atype[1])))
      except:
        gpus.append((atype[1], 1))  # didn't specify num
    elif len(atype) == 1 and atype[0] == "gpu":
        gpus.append(("gpu", 1))  # didn't specify gpu or num
    else:
      raise NotImplementedError
  return gpus


def parse_table(command: str, columns: List[str]) -> Iterable[Dict]:
  p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
  out = p.stdout.read().decode('utf-8')
  if FLAG['verbose']:
    print(f'COMMAND: {command}')
    print(out)
    print()
  rows = out.rstrip('\n')
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
  del gpu2status2user2count[UNK_GPU]
  print('{:<10} | {:<40}{:>8} | {:<35}{:>8} | {:>6} | {:>6}'.format('GPU', f'top-{max_display_user}', 'Running', f'top-{max_display_user}', 'Pending', 'Free', 'Total'))
  print('-' * 126)
  allgpu: Set[str] = set(gpu2count.keys())
  inusegpu: Set[str] = set(gpu2status2user2count.keys())
  for gpu in sorted(list(inusegpu) + list(allgpu - inusegpu)):
    status2user2count = gpu2status2user2count[gpu]
    status2count = defaultdict(lambda: 0)
    status2count.update({s: sum(u2c.values()) for s, u2c in status2user2count.items()})
    status2count['F'] = gpu2count[gpu] - status2count['R']
    print('{:<10} | {:<40}{:>8} | {:<35}{:>8} | {:>6} | {:>6}'.format(
      gpu,
      show_user(status2user2count[RUN_ST], 40), status2count[RUN_ST],
      show_user(status2user2count[PEND_ST], 35), status2count[PEND_ST],
      status2count['F'], gpu2count[gpu]))
    for u, c in status2user2count[RUN_ST].items():
      user2count[u] += c
  print(f"(unknown type GPU: {unknow_gpu[RUN_ST]} running {unknow_gpu[PEND_ST]} pending)")
  print('--------- top users ---------')
  for u, c in sorted(user2count.items(), key=lambda x: (-x[1], x[0]))[:max_users_total]:
    print('{:<15}\t{:10}'.format(u, c))


def get_gpu_config(filename: str = '/run/slurm/conf/nodes.conf') -> Tuple[Dict[str, Dict[int, str]], Dict[str, int]]:
  available_nodes_command = 'sinfo --responding -N -o "%N;%t" --noheader'
  p = subprocess.Popen(available_nodes_command, shell=True, stdout=subprocess.PIPE)
  all_nodes_and_states = set(
    tuple(n.split(";"))
    for n in p.stdout.read().decode('utf-8').strip().splitlines()
  )
  available_nodes = {
    n[0]
    for n in all_nodes_and_states
    if n[1] in ("idle", "mix", "alloc")
  }
  gpu2count: Dict[str, int] = defaultdict(lambda: 0)
  node2id2gpu: Dict[str, Dict[int, str]] = defaultdict(lambda: {})
  with open(filename, 'r') as fin:
    for l in fin:
      l = l.strip()
      if len(l) <= 0 or l[0] == '#':
        continue
      # Example: NodeName=babel-a-20 CPUs=128 Boards=1 SocketsPerBoard=2 CoresPerSocket=32 ThreadsPerCore=2 RealMemory=483000 Gres=gpu:A100_40GB:8,gpu:K80:1
      m = re.match(r"^NodeName=([-a-z0-9]+).*Gres=([:a-zA-Z0-9_,]+).*$", l)
      if not m:
        continue
      node, gpu_sets = m.groups()
      if node not in available_nodes:
        continue
      gpu_idx = 0
      for gpu_set in gpu_sets.split(","):
        gres_type, gpu_type, _gpu_count = gpu_set.split(":")
        gpu_count = int(_gpu_count)
        if gres_type != "gpu":
          continue
        gpu2count[gpu_type] += gpu_count
        for i in range(gpu_idx, gpu_idx + gpu_count):
          node2id2gpu[node][i] = gpu_type
        gpu_idx += gpu_count
  return node2id2gpu, gpu2count


def get_job_info(jobid: str) -> Dict[str, Any]:
  # Note: this won't work with job arrays, but job arrays' base job IDs are not listed when using "squeue -o %i" so we should be fine
  job_command = f'scontrol show jobid -dd {jobid}'
  gpu_anchor = ' GRES=gpu'
  node_anchor = ' Nodes='
  result = {'nodes': [], 'gpu_ids': []}
  p = subprocess.Popen(job_command, shell=True, stdout=subprocess.PIPE)
  out = p.stdout.read().decode('utf-8')
  if FLAG['verbose']:
    print(f'COMMAND: {job_command}')
    print(out)
    print('')
  job_info = out.rstrip('\n')
  # Example line to parse: Nodes=babel-1-27 CPU_IDs=0-15,32-47 Mem=204800 GRES=gpu:A6000:4(IDX:0-3)
  # TODO: there are multiple such lines in multi-node job, currently we are only parsing the first one
  n = job_info.find(node_anchor)
  nodes = parse_nodes(job_info[n + len(node_anchor):].split(' ', 1)[0])
  g = job_info.find(gpu_anchor)
  if g == -1:
    return result
  ids: str = job_info[g + len(gpu_anchor):].split('(', 1)[1].split(')', 1)[0][len('IDX:'):]
  if ',' in ids:  # separate ids
    _ids: List[str] = ids.split(',')
  elif ids.strip() == '':  # skip this job
    return result
  else:
    _ids: List[str] = [ids]
  merge_ids: List[int] = []
  for __ids in _ids:
    if '-' in __ids:  # a range of ids
      s, e = __ids.split('-', 1)
      merge_ids.extend(range(int(s), int(e) + 1))
    else:
      merge_ids.append(int(__ids))
  # TODO: the output structure will need to be modified if we want to support multi-node jobs
  result['nodes'] = [nodes[0]]
  result['gpu_ids'] = merge_ids
  return result


def gpu_summary():
  info_cols = ['NODELIST', 'CPUS', 'MEMORY', 'AVAIL_FEATURES', 'GRES']
  info_command = 'sinfo -a -o "%50N\t%10c\t%10m\t%25f\t%50G"'
  job_cols = ['JOBID', 'PARTITION', 'USER', 'ST', 'TIME', 'NODES', 'NODELIST(REASON)', 'NAME', 'TRES_PER_NODE']
  job_command = 'squeue -a -o "%.18i\t%.9P\t%.20u\t%.2t\t%.12M\t%.6D\t%.25R\t%.20j\t%.20b"'

  # summarize gpu
  node2id2gpu, gpu2count = get_gpu_config()

  # summarize jobs
  gpu2status2user2count: Dict[str, Dict[str, Dict[str, int]]] = \
    defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
  for job in parse_table(job_command, job_cols):
    user = job['USER']
    jobid = job['JOBID']
    # TODO: fix this, TRES_PER_NODE is "N/A" for multi-node GPU jobs, so parse_gres() returns empty list
    gpus = parse_gres(job['TRES_PER_NODE'])
    reason = job["NODELIST(REASON)"]
    st = job['ST']
    if st == "PD" and reason not in PENDING_REASONS:
      st = "PD-other"
    if st == RUN_ST and parse_nodes(job['NODELIST(REASON)'])[0] not in node2id2gpu:
      continue
    for gpu in gpus:
      gputype, count = gpu
      if count <= 0:
        continue
      if gputype == UNK_GPU and st == RUN_ST:
        jobinfo = get_job_info(jobid)
        if len(jobinfo['nodes']) <= 0:
          continue
        ids = jobinfo['gpu_ids'][:count]
        node = jobinfo['nodes'][0]
        for i in ids:
          gpu2status2user2count[node2id2gpu[node][i]][st][user] += 1
        count -= len(ids)
      if count > 0:
        gpu2status2user2count[gputype][st][user] += count

  # display
  pretty(gpu2status2user2count, gpu2count, max_display_user=3)


parser = argparse.ArgumentParser(description='slurm gpu info')
parser.add_argument('--verbose', action='store_true', help='print results of all commands used')
args = parser.parse_args()
FLAG['verbose'] = args.verbose
gpu_summary()
