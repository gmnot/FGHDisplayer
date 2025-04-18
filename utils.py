from collections import defaultdict
from datetime import datetime
import functools
from functools import wraps
import inspect
import os
import time
from typing import Any, List

def timed(f, *args, **kwargs):
  start = time.perf_counter()
  ret = f(*args, **kwargs)
  duration = time.perf_counter() - start
  return [duration, ret]

def track_total_time():
  total_time = {"value": 0}

  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      duration, result = timed(func, *args, **kwargs)
      total_time["value"] += duration
      return result

    wrapper.get_total_time = lambda: total_time["value"]
    return wrapper

  return decorator

def print_total_time(func):
  print(f"{func.__name__}: {func.get_total_time():.2f} sec in total")

def not_none(a):
  assert a is not None
  return a

def get_file_mtime_str(path):
  ts = os.path.getmtime(path)
  dt = datetime.fromtimestamp(ts)
  return dt.strftime('%H:%M:%S + %Y-%m-%d')

_trace_calls_depths_by_f : defaultdict[str, int] = defaultdict(int)

# examples:
# @trace_calls(enabled=True)
# def factorial(n):
#   if n == 0:
#     return 1
#   return n * factorial(n - 1)

# @trace_calls(enabled=False)
# def fib(n):
#   if n <= 1:
#     return n
#   return fib(n - 1) + fib(n - 2)

def trace_calls(enabled=True):
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      fname = func.__name__

      if enabled:
        _trace_calls_depths_by_f[fname] += 1
        depth = _trace_calls_depths_by_f[fname]
        indent = ' ' * (depth * 2)

        # Get argument names and values
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        arg_str = ', '.join(f"{k}={v!r}" for k, v in bound_args.arguments.items())

        if depth == 1:
          # Print left paren for non-recursive calls in the same line
          print(f"{indent}{fname}({arg_str}", end='')
        else:
          print(f"{indent}{fname}({arg_str}")

      try:
        result = func(*args, **kwargs)
        return result
      finally:
        if enabled:
          depth = _trace_calls_depths_by_f[fname]
          indent = ' ' * (depth * 2)

          # Check if we're at the last level of recursion
          if depth == 1:
            print(")")  # Close the inline parentheses for non-recursive calls
          else:
            # For recursive calls, print the closing parenthesis on a new line and keep indenting
            print(f"{indent})")

          _trace_calls_depths_by_f[fname] -= 1

    return wrapper
  return decorator

def validate_return_based_on_arg(arg_name, check_fn):
  def decorator(func):
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      bound = sig.bind(*args, **kwargs)
      bound.apply_defaults()
      arg_value = bound.arguments.get(arg_name)

      result = func(*args, **kwargs)

      values = result if isinstance(result, tuple) else (result,)
      for val in values:
        if not check_fn(val, arg_value):
          raise ValueError(f"Invalid return value: {val} ({arg_name}={arg_value})")

      return result
    return wrapper
  return decorator
# example:
# @validate_return_based_on_arg('mode', lambda ret, mode: ret > 0 if mode == 'strict' else True)
# def get_score(data, mode='relaxed'):
#   if not data:
#     return -1
#   return len(data)

class Recorder:
  FAIL = 0  # nothing changed, can't calc
  SUCC = 1  # changed
  DONE = 2  # changed (so it's a kind of SUCC), and know can't continue

  rec_limit : int
  cal_limit : int
  data      : List  # allow extra elem for result
  done      : bool  # mathmatically done, can't continue
  n_discard : int
  n_pre_discard : int  # discarded before reach limit
  will_skip_next : bool
  can_override_last : bool  # last not important, can override

  until     : Any
  until_met : bool

  def __init__(self, rec_limit, cal_limit, until=None):
    assert rec_limit > 0
    self.rec_limit = rec_limit
    self.cal_limit = cal_limit
    self.data      = []
    self.done      = False
    self.n_discard = 0
    self.n_pre_discard = 0
    self.will_skip_next = False
    self.can_override_last = False
    self.until = until
    self.until_met = False

  def __str__(self):
    if self.cnt() == 0:
      return '[]'
    last = f'{self.data[-1]}'
    if len(last) > 30:
      return last
    return f'{self.data[::-1]}'

  def __repr__(self):
    return self.__str__()

  def cnt(self) -> int:
    return len(self.data)

  def tot_discard(self):
    return self.n_discard + self.n_pre_discard

  def tot_calc(self):
    return len(self.data) + self.tot_discard()

  def cal_limit_reached(self):
    return self.tot_calc() >= self.cal_limit

  def no_mid_steps(self):
    return self.rec_limit == 2

  def _record(self, entry):
    if self.can_override_last:
      self.data[-1] = entry
      self.can_override_last = False
    elif self.cnt() >= self.rec_limit:
      self.n_discard += 1
      self.data[-1] = entry
    else:
      self.data.append(entry)

  # return: True if cal_limit_reached or until met
  def record(self, entry) -> bool:
    assert entry is not None
    try:
      if self.until is not None and \
         entry == self.until:
        self.until_met = True
      self._record(entry)
      return self.inc_discard_check_end()

    finally:
      if self.will_skip_next:
        self.will_skip_next = False
        self.can_override_last = True
      else:
        self.can_override_last = False

  def get_result(self):
    assert len(self.data) > 0
    return self.data[-1]

  # return True if cal_limit reached
  def inc_discard_check_end(self, *, n_steps=1) -> bool:
    if self.cnt() >= self.rec_limit:
      self.n_discard += n_steps
    else:
      self.n_pre_discard += n_steps
    return self.cal_limit_reached() or self.until_met

  def skip_next(self):
    self.will_skip_next = True

  # todo 3: arg for one line mode
  def to_latex(self, entry_to_latex):
    ret = r' \begin{align*}' + '\n'
    ret += entry_to_latex(self.data[0])
    for ord in self.data[1:-1]:
      ret += f'  &= {entry_to_latex(ord)} \\\\\n'
    if self.n_discard > 1:  # 1 step could be (a+b)[x] = a+b[x]
      ret += r'  &\phantom{=} \vdots \quad \raisebox{0.2em}{\text{' + \
             f'after {self.n_discard} more steps' r'}} \\' + '\n'
    ret += f'  &= {entry_to_latex(self.data[-1])} '
    if not self.done:
      ret += r' = \dots'
    ret += '\\\\\n'
    ret += r'\end{align*} ' + '\n'
    return ret
