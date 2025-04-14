from datetime import datetime
import time
from functools import wraps
import os

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
