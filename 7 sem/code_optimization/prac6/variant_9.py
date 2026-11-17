"""Три модели выполнения; исходные этапы и эталон не изменены."""
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import get_context
import time
from variant_9_baseline import (
    stage_fetch, stage_fetch_async, stage_cpu, stage_post,
    run_sequential, make_workload, SCALES,
)

WORKERS = 4


def fetch_and_cpu(job):
    # В процесс передаётся только номер задания, обратно — одно число.
    return stage_cpu(stage_fetch(job))


def run_threads(jobs):
    state = 0.0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for value in pool.map(fetch_and_cpu, jobs):
            state = stage_post(value, state)
    return state, len(jobs)


def run_processes(jobs):
    state = 0.0
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn")) as pool:
        for value in pool.map(fetch_and_cpu, jobs):
            state = stage_post(value, state)
    return state, len(jobs)


async def _run_async(jobs):
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn")) as pool:
        async def prepare(job):
            data = await stage_fetch_async(job)
            return await loop.run_in_executor(pool, stage_cpu, data)

        values = await asyncio.gather(*(prepare(job) for job in jobs))
        state = 0.0
        for value in values:
            # state зависит от предыдущего шага; порядок нельзя менять.
            state = await loop.run_in_executor(pool, stage_post, value, state)
    return state, len(jobs)


def run_async(jobs):
    return asyncio.run(_run_async(jobs))


def run_task(scale=4):
    jobs = make_workload(scale)
    start = time.perf_counter()
    result = run_processes(jobs)
    return round(result[0], 4), result[1], round(time.perf_counter() - start, 4)


if __name__ == "__main__":
    for scale in SCALES:
        print(f"scale={scale}  {run_task(scale)}")
