import threading
import queue

from simpl.collector import BaseWorker


class Worker(BaseWorker):
    def __init__(self, collector):
        self.collector = collector

    def collect_episode(self, *args, **kwargs):
        episode = self.collector.collect_episode(*args, **kwargs)
        return episode


class MultiThreadCollector:
    def __init__(self, workers):
        self.work_queue = queue.SimpleQueue() 
        self.received_queue = queue.SimpleQueue() 
        self.result_queue = queue.SimpleQueue() 
        
        self.processes = []
        for worker in workers:
            queues = (self.work_queue, self.received_queue, self.result_queue)
            p = threading.Thread(target=worker, args=queues)
            p.start()
            self.processes.append(p)
        self.work_i = 0
    
    def submit(self, *args, **kwargs):
        msg = (self.work_i, args, kwargs)
        self.work_queue.put(msg)
        self.received_queue.get()
        
        self.work_i += 1
    
    def wait(self):
        episodes = [None]*self.work_i
        while self.work_i > 0:
            msg = self.result_queue.get()
            work_i, episode = msg
            episodes[work_i] = episode
            self.work_i -= 1
        return episodes
    
    def close(self):
        for _ in range(len(self.processes)):
            self.work_queue.put(False)
        for process in self.processes:
            process.join()
