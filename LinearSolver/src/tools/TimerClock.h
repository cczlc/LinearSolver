#ifndef CPP11TIMER_TIMERCLOCK_H
#define CPP11TIMER_TIMERCLOCK_H

#include <chrono>

class TimerClock {
public:
    TimerClock();
    virtual ~TimerClock() = default;
    void update();
    double getSecond();
    double getMilliSecond();
    double getMicroSecond();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
};


#endif //CPP11TIMER_TIMERCLOCK_H