#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <utils.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace nebula {

class ThreadPool : NonMovable {

    public:
        static ThreadPool& instance();

        ~ThreadPool();

        template<typename F>
        void parallelFor(u32 count, F&& fn) {
            if(count == 0) {
                return;
            }
            // If the work is small, run it sequentially to avoid the overhead of threading
            constexpr u32 minParallel = 64;
            if(count < minParallel || _workers.empty()) {
                for(u32 i = 0; i != count; ++i) {
                    fn(i);
                }
                return;
            }

            auto thunk = [](void* ctx, u32 begin, u32 end) {
                F& f = *static_cast<F*>(ctx);
                for(u32 i = begin; i != end; ++i) {
                    f(i);
                }
            };
            run(count, thunk, &fn);
        }

    private:
        ThreadPool();

        void workerLoop();
        void drain();
        void run(u32 count, void (*fn)(void*, u32, u32), void* ctx);

        std::mutex _mutex;
        std::condition_variable _workCv;
        std::condition_variable _doneCv;

        void (*_fn)(void*, u32, u32) = nullptr;
        void* _ctx = nullptr;
        u32 _count = 0;
        u32 _grain = 64;
        std::atomic<u32> _next{0};
        std::atomic<u32> _remaining{0};
        bool _jobActive = false;
        bool _stop = false;

        std::vector<std::thread> _workers;
};

template<typename F>
void parallelFor(u32 count, F&& fn) {
    ThreadPool::instance().parallelFor(count, std::forward<F>(fn));
}

}

#endif // THREADPOOL_H
