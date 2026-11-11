#include <libproc.h>
#include <sys/resource.h>
#include <stdint.h>

uint64_t stt_phys_footprint_peak(int pid) {
    struct rusage_info_v4 info = {0};
    if (proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t *)&info) != 0) {
        return 0;
    }
    return info.ri_lifetime_max_phys_footprint;
}
