from contextlib import contextmanager
from time import perf_counter

def _noop_logger(msg: str): ...

@contextmanager
def step_timer(name: str, logger=None):
    log = logger or _noop_logger
    t0 = perf_counter()
    log.info(f"▶ {name} ...")
    yield lambda: perf_counter() - t0   # devolve duração ao sair
    log.info(f"✔ {name} concluído")
