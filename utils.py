from collections import defaultdict
from datetime import datetime
import functools
from functools import wraps
import inspect
import os
import time


def track_total_time():
  total_time = {"value": 0}

  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      start = time.perf_counter()
      result = func(*args, **kwargs)
      duration = time.perf_counter() - start
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
