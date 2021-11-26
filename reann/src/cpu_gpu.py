#作者：Yuwei
#链接：https://www.zhihu.com/question/307282137/answer/1560137140
#来源：知乎
from threading import Thread
import torch
from queue import Queue

class CudaDataLoader:
    """ 异步预先将数据从CPU加载到GPU中 """

    def __init__(self, loader, device, queue_size=0):
        self.device = device
        self.loader = loader
        self.idx = 0
        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=queue_size)
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()
        self.length=self.loader.length

    def load_loop(self):
        """ 不断的将cuda数据加载到队列里 """
        # The loop that will load into the queue in the background
        while True:
            with torch.cuda.stream(self.load_stream):
                for sample in self.loader:
                    self.queue.put(self.load_instance(sample))

    def load_instance(self, sample):
        """ 将batch数据从CPU加载到GPU中 """
        if torch.is_tensor(sample):
            return sample.to(self.device, non_blocking=True)
        else:
            return (self.load_instance(s) for s in sample)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        # 加载线程挂了
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # 一个epoch加载完了
        elif self.idx >= self.length:
            self.idx = 0
            raise StopIteration
        # 下一个batch
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

