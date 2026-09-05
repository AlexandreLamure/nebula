#include "ThreadPool.h"

#include <algorithm>

namespace nebula {

ThreadPool& ThreadPool::instance() {
    static ThreadPool pool;
    return pool;
}

ThreadPool::ThreadPool() {
    const u32 hw = std::max(1u, std::thread::hardware_concurrency());
    const u32 workerCount = hw - 1;
    _workers.reserve(workerCount);
    for(u32 i = 0; i != workerCount; ++i) {
        _workers.emplace_back([this] { workerLoop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard lock(_mutex);
        _stop = true;
    }
    _workCv.notify_all();
    _doneCv.notify_all();
    for(std::thread& worker : _workers) {
        worker.join();
    }
}

void ThreadPool::drain() {
    for(;;) {
        const u32 begin = _next.fetch_add(_grain, std::memory_order_relaxed);
        if(begin >= _count) {
            return;
        }
        const u32 end = std::min(begin + _grain, _count);
        _fn(_ctx, begin, end);
    }
}

void ThreadPool::workerLoop() {
    for(;;) {
        {
            std::unique_lock lock(_mutex);
            _workCv.wait(lock, [this] { return _stop || _jobActive; });
            if(_stop) {
                return;
            }
        }

        drain();

        if(_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard lock(_mutex);
            _jobActive = false;
            _doneCv.notify_all();
        } else {
            std::unique_lock lock(_mutex);
            _doneCv.wait(lock, [this] { return !_jobActive || _stop; });
            if(_stop) {
                return;
            }
        }
    }
}

void ThreadPool::run(u32 count, void (*fn)(void*, u32, u32), void* ctx) {
    const u32 threads = u32(_workers.size()) + 1;
    const u32 grain = std::max(32u, (count + threads * 4 - 1) / (threads * 4));

    {
        std::lock_guard lock(_mutex);
        _fn = fn;
        _ctx = ctx;
        _count = count;
        _grain = grain;
        _next.store(0, std::memory_order_relaxed);
        _remaining.store(threads, std::memory_order_relaxed);
        _jobActive = true;
    }
    _workCv.notify_all();

    drain();

    if(_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard lock(_mutex);
        _jobActive = false;
        _doneCv.notify_all();
        return;
    }

    std::unique_lock lock(_mutex);
    _doneCv.wait(lock, [this] { return !_jobActive; });
}

}
