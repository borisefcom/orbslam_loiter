#ifndef ORB_SLAM3_OS_UTILS_H
#define ORB_SLAM3_OS_UTILS_H

#ifdef _WIN32
#include <chrono>
#include <thread>

inline void usleep(unsigned int usec)
{
    std::this_thread::sleep_for(std::chrono::microseconds(usec));
}
#else
#include <unistd.h>
#endif

#endif // ORB_SLAM3_OS_UTILS_H

