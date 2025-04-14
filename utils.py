import time
from functools import wraps

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
