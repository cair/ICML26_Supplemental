import time
import pynvml
import os
import threading


class Monitor:
    start_time: float
    end_time: float
    peak_gpu_memory: float

    def __init__(self, gpu_id, gpu_polling_rate: float = 0.05):
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        self.pid = os.getpid()
        self.monitoring = True
        self.peak_gpu_memory = 0.0
        self.gpu_polling_rate = gpu_polling_rate

    def _monitor_gpu_memory(self):
        while self.monitoring:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.gpu_handle)
            used_mem = 0.0
            for p in procs:
                if p.pid == self.pid:
                    used_mem = p.usedGpuMemory / (1024 ** 2)
            self.peak_gpu_memory = max(self.peak_gpu_memory, used_mem)
            time.sleep(self.gpu_polling_rate)

    def __enter__(self):
        self.start_time = time.time()
        self.peak_gpu_memory = 0.0
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_gpu_memory, daemon=True)
        self.monitor_thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.monitoring = False
        self.monitor_thread.join(timeout=1)

    def elapsed(self):
        if self.end_time is None:
            raise RuntimeError("elapsed() must be called after context is ended.")
        return self.end_time - self.start_time

    def peak_memory(self):
        return self.peak_gpu_memory
