class Monitor:
    def __init__(self,folder):
        self.folder=folder;self.stop=threading.Event();self.error=None;self.samples=0;self.peak_private=0
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def _observe(self):
        from resource_watchdog_v2 import sample_memory
        s=sample_memory(os.getpid(),'resident_RAM_system_headroom');self.samples+=1
        require(s['paired_sampling_within_one_second'],'Fresh actual process/system memory')
        self.peak_private=max(self.peak_private,s['worker_private_memory']['private_bytes'])
        m=s['system_memory']
        if m['commit_available_bytes']<2000000000 or m['physical_available_bytes']<1500000000:
            write_json(self.folder/'actual_failing_headroom_sample.json',s)
        require(m['commit_available_bytes']>=2000000000 and m['physical_available_bytes']>=1500000000,
            'Sensible real system headroom; no obsolete2GB CPU-family or90percent prospective ceiling')
        return s
    def _run(self):
        try:
            while not self.stop.wait(.5):self._observe()
        except BaseException as e:self.error=e
    def check(self):
        if self.error is not None:raise RuntimeError('Actual resident memory monitor: '+repr(self.error))
        require(not (self.folder/'stop.json').exists(),'Actual owner cooperative stop')
    def close(self):
        self.stop.set();self.thread.join(timeout=3);require(not self.thread.is_alive(),'Actual RAM monitor joined')
        write_json(self.folder/'monitor_completion.json',{'samples':self.samples,'maximum_sampled_private_bytes':self.peak_private,
            'joined':True,'clean':self.error is None,'error':None if self.error is None else repr(self.error)})
