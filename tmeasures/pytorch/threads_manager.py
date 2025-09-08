import abc
import threading
from typing import Any, Callable

from tmeasures.utils.iterable_queue import FullQueue, IterableQueue

from .. import logger as tm_logger

logger = tm_logger.getChild("pytorch.threads_manager")

import concurrent.futures
import queue


class ComputationModel(abc.ABC):

    def __init__(self,server_function:Callable,worker_functions:dict[str,Callable],rows:int,cols:int,batch_size:int) -> None:
        self.server_function = server_function
        self.worker_functions = worker_functions
        self.batch_calculator = RowBatchCalculator(rows,cols,batch_size)
    @abc.abstractmethod
    def execute(self,server_,prefix="tm"):
        pass

    @abc.abstractmethod
    def put(self,worker:str,row:int,x:Any):
        pass

class RowBatchCalculator():
    def __init__(self,rows:int,cols:int,batch_size:int) -> None:
        self.rows=rows
        self.cols=cols
        self.batch_size = batch_size
        logger.info(f"rows {rows} cols {cols} bs {batch_size}")
        self.reset()

    def reset(self):
        self.previous_batch_remanent= 0
        self.row_batches=None

    def update_to_row_formula(self,row:int):
        cols,batch_size = self.cols,self.batch_size
        start = row * cols
        end = start + cols - 1
        self.row_batches =  end // batch_size - start // batch_size + 1
        # logger.info(f"RBC: row {row}, {formula_row_batches} formula row batches")

    def update_to_row(self,row:int):
        if self.previous_batch_remanent >= self.cols:
            self.previous_batch_remanent -= self.cols
            self.row_batches = 1
        else:
            self.row_batches=0
            # if there are previous row remanents, there will be a batch for these
            if self.previous_batch_remanent>0:
                self.row_batches+=1
            # discount cols that were computed in a previous batch
            actual_cols = self.cols-self.previous_batch_remanent

            # "normal" number of batches
            self.row_batches += actual_cols // self.batch_size

            # add an extra batch if the actual cols is not divisible by batch_size
            extra = actual_cols % self.batch_size
            if extra>0:
                self.row_batches+=1
            # compute how many "cols" remain for the next row(s)
            # total batch size - cols for this row for the last batch
            self.previous_batch_remanent = max(0,self.batch_size-extra)
        # logger.info(f"RBC: row {row}, {self.row_batches} iterative row batches")

class ThreadsManager(ComputationModel):

    def __init__(self,server_function:Callable,worker_functions:dict[str,Callable],max_workers:int,rows:int,cols:int,batch_size:int) -> None:
        super(ThreadsManager, self).__init__(server_function,worker_functions,rows,cols,batch_size)
        self.stop=False
        self.max_workers = max_workers
        self.reset()

    def reset(self):
        names = self.worker_functions.keys()
        rows = self.batch_calculator.rows
        self.qs = {n: IterableQueue(rows,blocking_size=1,name=f"q({n})") for n in names}
        self.last_row = 0
        self.batch_calculator.reset()
        self.reset_row(0)

    def reset_row(self,row):
        self.batch_calculator.update_to_row(row)
        names = self.worker_functions.keys()
        batch_n = self.batch_calculator.row_batches
        self.row_qs = {n: IterableQueue(batch_n,blocking_size=1,name=f"q_row({n})") for n in names}

        for q,row_q in zip(self.qs.values(),self.row_qs.values()):
            q.put(row_q)
        self.row_batch = 0

    def put(self,row:int,x:dict[str,Any]):

        if self.last_row!=row:
            assert self.last_row+1==row
            self.last_row=row
            self.reset_row(row)
        # logger.info(f"putting row {row} (batch {self.row_batch}/{self.batch_calculator.row_batches}) to workers")
        self.row_batch+=1
        for worker_name,q in self.row_qs.items():

            # logger.info(f"put {worker_name}, row: {row}/{self.batch_calculator.rows}, queue {q.added}/{q.n}/{q.removed}, shape {x[worker_name].shape}")
            q.put(x[worker_name])

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
        worker_failure = any([e is not None for e in worker_exceptions])
        if server_exception is None and not worker_failure:
            logger.info("No exceptions found, threads finished.")
            return None
        else: # we have some exceptions
            # signal stop for server
            logger.info("Some threads failed, terminating")
            self.cancel(server_future)
            done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            if server_exception is not None:
                logger.info(f"Server exception, about to re raise from main thread\n{server_exception}\n")
                raise server_exception
            else:
                for e in worker_exceptions:
                    if e is not None:
                        logger.info(f"Worker exception, about to re raise from main thread\n{e}\n thread id {threading.get_ident()}\n")
                        raise e

