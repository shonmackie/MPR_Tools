"""
Parallelization function.
It feels weird to have a whole file just for this, but I need to be able to import it into both conversion_foil.py and spectrometer.py.
"""
import time
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor, Future
import multiprocessing as mp
from multiprocessing import SimpleQueue
from typing import Optional

from tqdm import tqdm


def run_concurrently(
        function: Callable,
        args_list: list[tuple],
        executor: Optional[Executor],
        progress_counter_total: float,
        task_title: str,
) -> list:
    """
    Run the given function many times in multiple different processes.
    A TQDM progress bar will be used to report on its progress.
    
    A single function will be executed in each process with a different set of arguments.
    The return values from all of the function calls will be returned as a list.
    The function should take the progress_counter and progress_lock as keyword arguments,
    and periodically increment progress_counter like so:
    
        with progress_lock:
            progress_counter.value += 1
    
    Parameters:
        function: the function to run.  as arguments, it should take the contents of each item in arg_lists,
                  plus a multiprocessing ValueProxy and a multiprocessing Lock
        args_list: the args to pass to the function in each process
        executor: the pool of workers to use.  if None, we will instantiate a new executor
        progress_counter_total: the value of the multiprocessing ValueProxy that should be considered 100%
        task_title: the label to put on the progress bar
        
    Returns:
        the result from calling function on each set of args, in a list
    """
    # Initialize executor, if we don't already have one
    if executor is not None:
        we_must_close_the_executor = False
    else:
        if len(args_list) > 1:
            executor = ProcessPoolExecutor(max_workers=len(args_list))
            we_must_close_the_executor = True
        else:
            executor = SerialExecutor()
            we_must_close_the_executor = False
    
    if type(executor) is not SerialExecutor:
        # Create shared counter for progress tracking
        manager = mp.Manager()
        progress_counter = manager.Value('i', 0)
        progress_lock = manager.Lock()
        # Initialize progress bar
        pbar = tqdm(total=progress_counter_total, desc=task_title)
    else:
        # Create dummy variables to fill in for the progress bar stuff
        progress_counter = FakeCounter(0)
        progress_lock = FakeLock()
        pbar = FakeProgressBar()
    
    try:
        futures = []
        
        # Submit jobs
        for worker_args in args_list:
            futures.append(
                executor.submit(
                    function, *worker_args,
                    progress_counter=progress_counter,
                    progress_lock=progress_lock))
        
        # Monitor progress while processes run
        last_count = 0
        while any(not future.done() for future in futures):
            current_count = progress_counter.value
            if current_count > last_count:
                pbar.update(current_count - last_count)
                last_count = current_count
            time.sleep(0.5)  # Check every x ms
        
        # Final update for any remaining progress
        final_count = progress_counter.value
        if final_count > last_count:
            pbar.update(final_count - last_count)
        
        # Collect results
        results = []
        for future in futures:
            results.append(future.result())
    
    finally:
        pbar.close()
        
        if we_must_close_the_executor:
            executor.shutdown()
    
    return results


class SerialExecutor(Executor):
    """ an Executor for when you don't actually want to do multiprocessing """
    def __init__(self):
        self.tasks = SimpleQueue()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
    
    def shutdown(self, wait=True, cancel_futures=True):
        pass
    
    def submit(self, function, *args, **kwargs):
        return OnDemandFuture(function, *args, **kwargs)


class OnDemandFuture(Future):
    """ a return value that pretends to be concurrent but actually isn't """
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
    
    def done(self):
        return True
    
    def result(self, timeout=None):
        return self.function(*self.args, **self.kwargs)


class FakeLock:
    """ for when you wrote your code to have a lock but you don't actually need a lock """
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class FakeCounter:
    """ for when you wrote your code to have a counter but you don't actually need a counter """
    def __init__(self, value: float):
        self.value = value


class FakeProgressBar:
    """ for when you wrote your code to have a progress bar but progress bars don't actually work without multiprocessing """
    def __init__(self):
        pass
    
    def update(self, value):
        pass
    
    def close(self):
        pass
