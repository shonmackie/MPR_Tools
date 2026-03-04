"""
Parallelization function.
It feels weird to have a whole file just for this, but I need to be able to import it into both conversion_foil.py and spectrometer.py.
"""
import time
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
import multiprocessing as mp
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
    # Create shared counter for progress tracking
    manager = mp.Manager()
    progress_counter = manager.Value('i', 0)
    progress_lock = manager.Lock()
    
    # Initialize progress bar
    pbar = tqdm(total=progress_counter_total, desc=task_title)
    
    # Initialize executor, if we don't already have one
    if executor is None:
        executor = ProcessPoolExecutor(max_workers=len(args_list))
        we_must_close_the_executor = True
    else:
        we_must_close_the_executor = False
    
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
