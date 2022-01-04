#include "TimerClock.h"

using namespace std::chrono;

TimerClock::TimerClock() {
    update();
}

void TimerClock::update() {
    mStart = high_resolution_clock::now();
}

double TimerClock::getSecond() {
    return getMicroSecond() * 0.000001;
}

double TimerClock::getMilliSecond() {
    return getMicroSecond() * 0.001;
}

double TimerClock::getMicroSecond() {
    return duration_cast<microseconds>(high_resolution_clock::now() - mStart).count();
}
