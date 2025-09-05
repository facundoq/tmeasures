import abc
import threading
from typing import Any, Callable
from tmeasures.utils.iterable_queue import FullQueue, IterableQueue

from .. import logger as tm_logger
logger = tm_logger.getChild("pytorch.threads_manager")

import concurrent.futures
import queue

class ComputationModel(abc.ABC):

    def __init__(self,server_function:Callable,worker_functions:dict[str,Callable],rows:int,n_batch:int,max_workers:int) -> None:
        self.server_function = server_function
        self.worker_functions = worker_functions
        self.max_workers = max_workers
        self.rows=rows
        self.n_batch=n_batch

    @abc.abstractmethod
    def execute(self,server_,prefix="tm"):
        pass
    
    @abc.abstractmethod
    def put(self,worker:str,row:int,x:Any):
        pass

class ThreadsManager(ComputationModel):

    def __init__(self,server_function:Callable,worker_functions:dict[str,Callable],max_workers:int,rows:int,n_batch:int) -> None:
        super(ThreadsManager, self).__init__(server_function,worker_functions,rows,n_batch,max_workers)
        self.stop=False
        names = self.worker_functions.keys()
        self.qs = {n: IterableQueue(rows,blocking_size=rows,name=f"q({n})") for n in names}
        self.row_qs = {n: IterableQueue(n_batch,blocking_size=1,name=f"q_row({n})") for n in names}
        self.last_row = {n:0 for n in names}
        for q,row_q in zip(self.qs.values(),self.row_qs.values()):
            for i in range(rows):
                q.put(row_q)
            
    
    def put(self,worker:str,row:int,x:Any):
        # last_row = self.last_row[worker]
        worker_row_q = self.row_qs[worker]
        logger.info(f"put {worker}, row: {row}/{self.rows}, queue {worker_row_q.added}/{worker_row_q.n}/{worker_row_q.removed}, shape {x.shape}")
        # if last_row!=row:
        #     if not worker_row_q.fully_consumed():
        #         worker_row_q.queue
        #     self.last_row[worker]=row
        #     worker_row_q.reset()
        # if worker_row_q.full() and not worker_row_q.fully_consumed():
        #     worker_row_q.queue.join()
        worker_row_q.put(x)

    def execute(self,prefix="tm"):
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers,thread_name_prefix=prefix) as executor:
            logger.info("starting server thread")
            server_thread = executor.submit(self.server_function,self)
            logger.info("starting worker threads")
            
            worker_threads = {l:executor.submit(self.worker_functions[l],q,l) for l,q in self.qs.items()}
            
            r = self.check_finished(worker_threads,server_thread)
            
            if r is None:
                return [t.result() for t in worker_threads.values()]
            else:
                raise r 
            
    @property
    def queues(self):
        return list(self.qs.values())+list(self.row_qs.values())

    def cancel(self,server_future):
        logger.info("Exception raised when computing measure, shutting down all computing threads...")
        self.stop=True
        for q in self.queues:
            q.stop=True
        logger.info(f"Emptying queues to ensure server can move on and stop..")
        self.empty_all()
        # wait for server thread to finish
        concurrent.futures.wait([server_future], return_when=concurrent.futures.ALL_COMPLETED)
        # threading.Event().wait(0.1)
        logger.info("Server stopped. Pushing values to queues to ensure workers can consume and stop..")
        self.stop_all()
        logger.info("Workers stopped")

    def empty_all(self):
        for q in self.queues:
            try: q.get(block=False)
            except queue.Empty: pass


    def stop_all(self):
        for q in self.queues:
            try: q.put(None,block=False)
            except queue.Full: pass
            except FullQueue: pass
    
    def check_finished(self,worker_futures,server_future):
        futures = [server_future]+list(worker_futures.values())
        logger.info(f"Waiting for {len(worker_futures)} worker(s) and server to finish...")
        done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
        logger.info(f"done")
        # check if an exception was raised
        
        if server_future in done:
            server_exception = server_future.exception()
        else:
            server_exception = None
        worker_exceptions = [f.exception() if f in done else None for f in worker_futures.values()]
        worker_failure = any([not e is None for e in worker_exceptions])
        if server_exception is None and not worker_failure:
            logger.info("No exceptions found, threads finished.")
            return None
        else: # we have some exceptions
            # signal stop for server
            logger.info("Some threads failed, terminating")
            self.cancel(server_future)
            done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            if not server_exception is None:
                logger.info(f"Server exception, about to re raise from main thread\n{server_exception}\n")
                raise server_exception
            else:
                for e in worker_exceptions:
                    if not e is None:
                        logger.info(f"Worker exception, about to re raise from main thread\n{e}\n thread id {threading.get_ident()}\n")
                        raise e
        
    