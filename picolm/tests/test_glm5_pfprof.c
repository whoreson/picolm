/* The prefill profiler must produce output without relying on atexit(): per-layer lines with
 * PICOLM_GLM5_PFPROF=2 and a summary from glm5_pfprof_print(). Checked through a captured stderr. */
#include "model_glm5.c"
#include <time.h>
#include <unistd.h>
double get_time_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6; }
void bench_emit(int l, int p, double e, long a, long b) { (void)l;(void)p;(void)e;(void)a;(void)b; }
static int has(const char *hay, const char *needle) { return strstr(hay, needle) != NULL; }
int main(void) {
    setenv("PICOLM_GLM5_PFPROF", "2", 1);
    char path[] = "/tmp/pfprof_test_XXXXXX"; int fd = mkstemp(path);
    fflush(stderr); int saved = dup(2); dup2(fd, 2);
    for (int il = 0; il < 2; il++) {
        const int kind = il & 1;
        PFP_BEGIN(tL); PFP_BEGIN(tA); usleep(2000); PFP_END(tA, kind, PF_HC_ATTN_PRE);
        PFP_BEGIN(tF); usleep(3000); PFP_END(tF, kind, PF_FFN);
        g_pf_ms[kind][PF_LAYER] += glm5_prof_now() - tL; g_pf_cur[PF_LAYER] = glm5_prof_now() - tL; g_pf_layers[kind]++;
        glm5_pfprof_layer_line(il, kind);
    }
    glm5_pfprof_print();
    fflush(stderr); dup2(saved, 2); close(saved);
    char buf[8192] = {0}; lseek(fd, 0, SEEK_SET); ssize_t n = read(fd, buf, sizeof buf - 1); (void)n; close(fd); unlink(path);
    int ok = has(buf, "PFPROF layer  0 KDA") && has(buf, "PFPROF layer  1 MLA") && has(buf, "unaccounted") &&
             has(buf, "KDA layers") && has(buf, "MLA layers") && has(buf, "ffn / moe") && g_pf_layers[0] == 0;
    printf("%s\n%s", ok ? "PASS" : "FAIL", ok ? "" : buf);
    return !ok;
}
