// generate — text generation on the KV260: prefill, then one decode program per new token, with
// the KV cache in DDR. sw/compiler/mini_tpu/session.py is the same loop on the simulator.
//
//   generate [-d dir] [-f fw.hex] [-t timeout_s] [-v]          serve requests on stdin
//   generate [-d dir] [-f fw.hex] [-t timeout_s] -c [-m pos]   decode check
//
// Requests, one per line, token ids in and out:
//   <max_new> <temperature> <top_k> <seed> <n_stop> <stop ids ...> <n> <prompt ids ...>
// Prints "R <vocab> <seq> <ctx>" once ready. For each request, one "T <id>" line per token as it
// is made, then "E <reason> <tokens> <prefill_ms> <ms_per_token>"; reason is stop (a stop id was
// made, and printed), max, ctx (context full) or error. temperature 0 takes the most likely
// token; top_k 0 samples from the whole vocabulary. Prompts longer than seq go through prefill
// for the first seq tokens and through decode for the rest.
//
// -c: the prompt and then the tokens of golden_ids.bin, one at a time, must give the logits in
// golden_decode.bin bit for bit at each position it lists (up to -m). Exit code 0 if so.
//
// The DMA buffer holds prefill, the decode programs, weights, params, input, output and the KV
// cache, in that order (session.layout).

#define _GNU_SOURCE
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "mtpu.h"
#include "mtpu_desc.h"

#ifdef __has_include
#if __has_include("fw_image.h")
#include "fw_image.h"
#define HAVE_FW_IMAGE 1
#endif
#endif

#define ALIGN       4096u
#define MAX_BUCKETS 16
#define LINE_BYTES  (1u << 18)

typedef struct {
    long long d_model, n_layer, vocab, vocab_pad, seq, ctx, kv_bytes;
    long long prefill_input, prefill_output, decode_input, decode_output;
    int bucket[MAX_BUCKETS], n_buckets;
    double s_embed;
} model_t;

typedef struct {
    model_t m;
    mtpu_dev_t dev;
    uint32_t prog_off[1 + MAX_BUCKETS];     // prefill, then decode by bucket
    uint32_t w_off, p_off, in_off, out_off, kv_off, total;
    FILE *embed;                            // wte rows, read as needed
    float *wpe, *scale;
    uint64_t timeout_us;
    uint64_t cycles, desc_count;            // of the last run
    mtpu_perf_t perf;
} gen_t;

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------
static const char *dir = "model";

static FILE *open_in(const char *name, const char *mode)
{
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, mode);
    if (!f) perror(path);
    return f;
}

static long long file_size(const char *name)
{
    char path[512];
    struct stat st;
    snprintf(path, sizeof path, "%s/%s", dir, name);
    if (stat(path, &st)) { perror(path); return -1; }
    return (long long)st.st_size;
}

static int read_into(const char *name, void *dst, size_t bytes)
{
    FILE *f = open_in(name, "rb");
    if (!f) return -1;
    size_t got = fread(dst, 1, bytes, f);
    fclose(f);
    if (got != bytes) { fprintf(stderr, "%s/%s: short read\n", dir, name); return -1; }
    return 0;
}

static int read_model(model_t *m)
{
    struct { const char *key; long long *v; } keys[] = {
        { "d_model", &m->d_model }, { "n_layer", &m->n_layer }, { "vocab", &m->vocab }, { "vocab_pad", &m->vocab_pad },
        { "seq", &m->seq }, { "ctx", &m->ctx }, { "kv_bytes", &m->kv_bytes }, { "prefill_input", &m->prefill_input },
        { "prefill_output", &m->prefill_output }, { "decode_input", &m->decode_input }, { "decode_output", &m->decode_output },
    };
    FILE *f = open_in("model.txt", "r");
    if (!f) return -1;
    memset(m, 0, sizeof *m);
    char line[512], key[64];
    while (fgets(line, sizeof line, f)) {
        int at;
        if (sscanf(line, "%63s %n", key, &at) != 1) continue;
        char *v = line + at, *end;
        if (!strcmp(key, "s_embed")) m->s_embed = strtod(v, NULL);
        else if (!strcmp(key, "buckets")) {
            for (long b = strtol(v, &end, 10); end != v && m->n_buckets < MAX_BUCKETS; b = strtol(v, &end, 10)) {
                m->bucket[m->n_buckets++] = (int)b;
                v = end;
            }
        } else {
            for (size_t i = 0; i < sizeof keys / sizeof keys[0]; i++)
                if (!strcmp(key, keys[i].key)) *keys[i].v = strtoll(v, NULL, 10);
        }
    }
    fclose(f);
    if (!m->d_model || !m->vocab || !m->seq || !m->ctx || !m->n_buckets || m->s_embed <= 0 ||
        m->bucket[m->n_buckets - 1] != m->ctx) {
        fprintf(stderr, "%s/model.txt: incomplete\n", dir);
        return -1;
    }
    return 0;
}

static void decode_name(char *name, size_t n, int bucket)
{
    snprintf(name, n, "decode_%d.bin", bucket);
}

static uint32_t aligned(long long bytes)
{
    return (uint32_t)((bytes + ALIGN - 1) / ALIGN * ALIGN);
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------
static int load(gen_t *g, const char *fw_hex)
{
    model_t *m = &g->m;
    if (read_model(m)) return -1;

    char name[64];
    long long sizes[1 + MAX_BUCKETS], wb = file_size("weights.bin"), pb = file_size("params.bin");
    if (wb < 0 || pb < 0) return -1;
    uint64_t top = 0;
    for (int i = 0; i <= m->n_buckets; i++) {
        if (i) decode_name(name, sizeof name, m->bucket[i - 1]);
        if ((sizes[i] = file_size(i ? name : "program.bin")) < 0) return -1;
        g->prog_off[i] = (uint32_t)top;
        top += aligned(sizes[i]);
    }
    g->w_off = (uint32_t)top; top += aligned(wb);
    g->p_off = (uint32_t)top; top += aligned(pb);
    g->in_off = (uint32_t)top; top += aligned(m->prefill_input > m->decode_input ? m->prefill_input : m->decode_input);
    g->out_off = (uint32_t)top; top += aligned(m->prefill_output > m->decode_output ? m->prefill_output : m->decode_output);
    g->kv_off = (uint32_t)top; top += aligned(m->kv_bytes);
    if (top > 0xFFFFFFFFu) { fprintf(stderr, "model needs %" PRIu64 " bytes of DMA buffer\n", top); return -1; }
    g->total = (uint32_t)top;

    static uint32_t image[MINI_TPU_IMEM_WORDS];
    int words = -1;
    if (fw_hex) {
        words = mtpu_read_hex(fw_hex, image, MINI_TPU_IMEM_WORDS);
    } else {
#ifdef HAVE_FW_IMAGE
        memcpy(image, mini_tpu_fw_image, sizeof mini_tpu_fw_image);
        words = MINI_TPU_FW_IMAGE_WORDS;
#else
        fprintf(stderr, "no firmware built in (fw_image.h not found at build time): pass -f fw.hex\n");
#endif
    }
    if (words < 0) return -1;

    mtpu_dev_t *d = &g->dev;
    if (mtpu_open(d, MTPU_CTRL_PHYS_DEFAULT, g->total)) {
        fprintf(stderr, "the model needs a DMA buffer of %.1f MB (make load UDMABUF_SIZE=...)\n", g->total / 1048576.0);
        return -1;
    }
    uint32_t id = mtpu_rd(d, MINI_TPU_R_HW_ID);
    if (id != MINI_TPU_HW_ID) {
        fprintf(stderr, "HW_ID 0x%08x, expected 0x%08x: is mini_tpu.bit loaded? (make load)\n", id, MINI_TPU_HW_ID);
        return -1;
    }
    if (mtpu_boot(d, image, words)) return -1;

    uint32_t arch = mtpu_rd(d, MINI_TPU_R_HW_ARCH);
    for (int i = 0; i <= m->n_buckets; i++) {
        if (i) decode_name(name, sizeof name, m->bucket[i - 1]);
        const char *file = i ? name : "program.bin";
        if (read_into(file, d->buf + g->prog_off[i], (size_t)sizes[i]) ||
            mtpu_check_header(d->buf + g->prog_off[i], file, arch, i ? (uint32_t)m->bucket[i - 1] : 0))
            return -1;
    }
    if (read_into("weights.bin", d->buf + g->w_off, (size_t)wb) || read_into("params.bin", d->buf + g->p_off, (size_t)pb))
        return -1;
    memset(d->buf + g->kv_off, 0, (size_t)m->kv_bytes);
    if (mtpu_sync(d, true, 0, g->total)) return -1;
    mtpu_set_region(d, MINI_TPU_REGION_WEIGHTS, d->buf_phys + g->w_off);
    mtpu_set_region(d, MINI_TPU_REGION_PARAMS, d->buf_phys + g->p_off);
    mtpu_set_region(d, MINI_TPU_REGION_INPUT, d->buf_phys + g->in_off);
    mtpu_set_region(d, MINI_TPU_REGION_OUTPUT, d->buf_phys + g->out_off);
    mtpu_set_region(d, MINI_TPU_REGION_KV, d->buf_phys + g->kv_off);

    // embed.bin: wte [vocab, d_model] then wpe [ctx, d_model], f32
    size_t D = (size_t)m->d_model, V = (size_t)m->vocab, P = (size_t)m->ctx;
    if (file_size("embed.bin") != (long long)((V + P) * D * 4) || file_size("logit_scale.bin") != (long long)(V * 4)) {
        fprintf(stderr, "embed.bin or logit_scale.bin: wrong size for model.txt\n");
        return -1;
    }
    g->wpe = malloc(P * D * 4);
    g->scale = malloc(V * 4);
    if (!g->wpe || !g->scale || !(g->embed = open_in("embed.bin", "rb"))) return -1;
    if (fseek(g->embed, (long)(V * D * 4), SEEK_SET) || fread(g->wpe, 4, P * D, g->embed) != P * D) {
        fprintf(stderr, "embed.bin: short read\n");
        return -1;
    }
    return read_into("logit_scale.bin", g->scale, V * 4);
}

static void print_error(const mtpu_dev_t *d, uint32_t status)
{
    mtpu_error_t e;
    mtpu_read_error(d, status, &e);
    fprintf(stderr, "STATUS 0x%08x %s: chunk %u, descriptor %u, unit %u, info 0x%08x 0x%08x\n", status, mtpu_err_name(e.code),
            e.chunk, e.desc, e.unit, e.info0, e.info1);
}

static int run(gen_t *g, int prog)
{
    uint32_t st = mtpu_exec(&g->dev, g->prog_off[prog], g->timeout_us, NULL);
    if (st == MTPU_RUN_FAILED) return -1;
    if (MINI_TPU_ST_STATE(st) != MINI_TPU_ST_DONE) {
        print_error(&g->dev, st);
        return -1;
    }
    mtpu_perf_t p;
    mtpu_read_perf(&g->dev, &p);
    g->cycles = p.cycles;
    g->desc_count = p.desc_count;
    g->perf = p;
    return 0;
}

// token + position embedding on the residual grid, as QuantModel.embed: round half to even
static int embed(gen_t *g, const int *ids, int n, int pos, int32_t *x)
{
    size_t D = (size_t)g->m.d_model;
    float row[4096];
    if (D > sizeof row / sizeof row[0]) return -1;
    for (int i = 0; i < n; i++) {
        if (fseek(g->embed, (long)((size_t)ids[i] * D * 4), SEEK_SET) || fread(row, 4, D, g->embed) != D) {
            fprintf(stderr, "embed.bin: short read\n");
            return -1;
        }
        const float *pe = g->wpe + (size_t)(pos + i) * D;
        for (size_t j = 0; j < D; j++) {
            double v = rint(((double)row[j] + (double)pe[j]) / g->m.s_embed);
            x[(size_t)i * D + j] = (int32_t)(v > 32767 ? 32767 : v < -32767 ? -32767 : v);
        }
    }
    return 0;
}

// Prefill of ids[0 .. n), n <= seq, padded with token 0. acc: i32 logits of the last token.
static int prefill(gen_t *g, const int *ids, int n, int32_t *acc)
{
    const model_t *m = &g->m;
    static int padded[4096];
    int S = (int)m->seq;
    if (S > 4096) return -1;
    for (int i = 0; i < S; i++) padded[i] = i < n ? ids[i] : 0;
    if (embed(g, padded, S, 0, (int32_t *)(g->dev.buf + g->in_off))) return -1;
    if (mtpu_sync(&g->dev, true, g->in_off, (uint32_t)m->prefill_input) || run(g, 0)) return -1;
    uint32_t row = g->out_off + (uint32_t)((n - 1) * m->vocab_pad * 4);
    mtpu_sync(&g->dev, false, row, (uint32_t)(m->vocab_pad * 4));
    memcpy(acc, g->dev.buf + row, (size_t)m->vocab_pad * 4);
    return 0;
}

// token at position pos, after positions 0 .. pos - 1 are in the KV cache. Returns its bucket.
static int step(gen_t *g, int token, int pos, int32_t *acc)
{
    const model_t *m = &g->m;
    int b = 0;
    while (m->bucket[b] <= pos) b++;
    int32_t *x = (int32_t *)(g->dev.buf + g->in_off), *mask = x + m->d_model;
    if (embed(g, &token, 1, pos, x)) return -1;
    for (int j = 0; j < m->ctx; j++) mask[j] = j <= pos ? 0 : INT32_MIN;
    if (mtpu_sync(&g->dev, true, g->in_off, (uint32_t)m->decode_input)) return -1;
    mtpu_set_region(&g->dev, MINI_TPU_REGION_KV_ROW, g->dev.buf_phys + g->kv_off + (uint64_t)pos * (uint64_t)m->d_model);
    if (run(g, 1 + b)) return -1;
    mtpu_sync(&g->dev, false, g->out_off, (uint32_t)(m->vocab_pad * 4));
    memcpy(acc, g->dev.buf + g->out_off, (size_t)m->vocab_pad * 4);
    return m->bucket[b];
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------
static double uniform(uint64_t *s)                 // xorshift64*, [0, 1)
{
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return (double)((*s * 0x2545F4914F6CDD1Dull) >> 11) / 9007199254740992.0;
}

static int sample(const gen_t *g, const int32_t *acc, double temperature, int top_k, uint64_t *rng)
{
    int V = (int)g->m.vocab;
    static int idx[50304];
    static double val[50304];
    if (V > (int)(sizeof idx / sizeof idx[0])) return 0;
    int best = 0;
    for (int i = 1; i < V; i++)
        if ((double)acc[i] * g->scale[i] > (double)acc[best] * g->scale[best]) best = i;
    if (temperature <= 0) return best;

    // the top_k largest logits, largest first; all of them if top_k is 0
    int k = top_k <= 0 || top_k > V ? V : top_k, n = 0;
    for (int i = 0; i < V; i++) {
        double l = (double)acc[i] * g->scale[i];
        if (k == V) { idx[n] = i; val[n++] = l; continue; }
        if (n == k && l <= val[k - 1]) continue;
        int j = n < k ? n++ : k - 1;
        while (j > 0 && val[j - 1] < l) { idx[j] = idx[j - 1]; val[j] = val[j - 1]; j--; }
        idx[j] = i;
        val[j] = l;
    }
    double top = (double)acc[best] * g->scale[best], sum = 0;
    for (int i = 0; i < n; i++) sum += (val[i] = exp((val[i] - top) / temperature));
    double r = uniform(rng) * sum;
    for (int i = 0; i < n; i++)
        if ((r -= val[i]) < 0) return idx[i];
    return idx[0];
}

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------
static int next_long(char **p, long *v)
{
    char *end;
    *v = strtol(*p, &end, 10);
    if (end == *p) return -1;
    *p = end;
    return 0;
}

static void finish(const char *reason, int tokens, double prefill_ms, double decode_ms)
{
    printf("E %s %d %.1f %.1f\n", reason, tokens, prefill_ms, tokens > 1 ? decode_ms / (tokens - 1) : 0.0);
    fflush(stdout);
}

static void serve(gen_t *g, int verbose)
{
    const model_t *m = &g->m;
    int V = (int)m->vocab, ctx = (int)m->ctx;
    char *line = malloc(LINE_BYTES);
    int *ids = malloc((size_t)ctx * sizeof *ids);
    unsigned char *stop = malloc((size_t)V);
    int32_t *acc = malloc((size_t)m->vocab_pad * 4);
    if (!line || !ids || !stop || !acc) return;
    printf("R %d %lld %d\n", V, m->seq, ctx);
    fflush(stdout);

    while (fgets(line, LINE_BYTES, stdin)) {
        char *p = line, *end;
        long max_new, top_k, seed, n_stop, n, v;
        memset(stop, 0, (size_t)V);
        int bad = next_long(&p, &max_new);
        double temperature = strtod(p, &end);
        bad |= end == p;
        p = end;
        bad |= next_long(&p, &top_k) | next_long(&p, &seed) | next_long(&p, &n_stop);
        for (long i = 0; !bad && i < n_stop; i++) {
            bad |= next_long(&p, &v) || v < 0 || v >= V;
            if (!bad) stop[v] = 1;
        }
        bad |= next_long(&p, &n) || n < 1 || n >= ctx || max_new < 1;
        for (long i = 0; !bad && i < n; i++) {
            bad |= next_long(&p, &v) || v < 0 || v >= V;
            if (!bad) ids[i] = (int)v;
        }
        if (bad) {
            fprintf(stderr, "bad request; expected: max_new temperature top_k seed n_stop stop... n ids... (n < %d)\n", ctx);
            finish("error", 0, 0, 0);
            continue;
        }

        uint64_t rng = (uint64_t)seed * 0x9E3779B97F4A7C15ull + 1, t0 = mtpu_now_us();
        int S = (int)m->seq, first = n < S ? (int)n : S, pos = first, made = 0;
        int err = prefill(g, ids, first, acc);
        for (; !err && pos < n; pos++) err = step(g, ids[pos], pos, acc) < 0;
        double prefill_ms = (mtpu_now_us() - t0) / 1000.0, decode_ms = 0;
        const char *reason = "error";
        while (!err) {
            int tok = sample(g, acc, temperature, (int)top_k, &rng);
            printf("T %d\n", tok);
            fflush(stdout);
            made++;
            if (stop[tok]) { reason = "stop"; break; }
            if (made == max_new) { reason = "max"; break; }
            if (pos >= ctx) { reason = "ctx"; break; }
            uint64_t t1 = mtpu_now_us();
            int bucket = step(g, tok, pos, acc);
            err = bucket < 0;
            decode_ms += (mtpu_now_us() - t1) / 1000.0;
            if (verbose && !err)
                fprintf(stderr, "position %d: decode_%d, %.2f M cycles, %" PRIu64 " descriptors\n", pos, bucket, g->cycles / 1e6, g->desc_count);
            pos++;
        }
        finish(reason, made, prefill_ms, decode_ms);
    }
    free(line); free(ids); free(stop); free(acc);
}

// ---------------------------------------------------------------------------
// Decode check
// ---------------------------------------------------------------------------
static int check(gen_t *g, int last_pos)
{
    const model_t *m = &g->m;
    int ctx = (int)m->ctx, vp = (int)m->vocab_pad;
    long long gb = file_size("golden_decode.bin");
    int *ids = malloc((size_t)ctx * sizeof *ids);
    int32_t *gold = gb > 8 ? malloc((size_t)gb) : NULL, *acc = malloc((size_t)vp * 4);
    if (!ids || !gold || !acc || read_into("golden_ids.bin", ids, (size_t)ctx * 4) || read_into("golden_decode.bin", gold, (size_t)gb))
        return 1;
    int n = gold[0], count = gold[1];
    const int *pos = gold + 2;
    const int32_t *rows = gold + 2 + count;
    if (gb != (long long)(2 + count + (long long)count * vp) * 4 || n < 1 || n > m->seq) {
        fprintf(stderr, "golden_decode.bin: wrong size for model.txt\n");
        return 1;
    }
    int last = last_pos >= 0 && last_pos < pos[count - 1] ? last_pos : pos[count - 1];
    printf("generate -c: %s, %.1f MB in the DMA buffer at 0x%" PRIx64 ", prompt %d, positions %d to %d\n", dir,
           g->total / 1048576.0, g->dev.buf_phys, n, n - 1, last);

    struct { double ms, cycles, busy[MINI_TPU_NUM_UNITS], wait[MINI_TPU_NUM_UNITS]; uint64_t descs; int runs; } per[MAX_BUCKETS] = { { 0 } };
    uint64_t diffs = 0;
    int checked = 0, c = 0;
    for (int p = n - 1; p <= last; p++) {
        uint64_t t0 = mtpu_now_us();
        int bucket = p == n - 1 ? (prefill(g, ids, n, acc) ? -1 : 0) : step(g, ids[p], p, acc);
        double ms = (mtpu_now_us() - t0) / 1000.0;
        if (bucket < 0) return 1;
        int b = 0;
        while (bucket && m->bucket[b] != bucket) b++;
        if (bucket) {
            per[b].ms += ms; per[b].cycles += (double)g->cycles; per[b].descs = g->desc_count; per[b].runs++;
            for (unsigned u = 0; u < MINI_TPU_NUM_UNITS; u++) {
                per[b].busy[u] += g->perf.busy[u];
                per[b].wait[u] += g->perf.wait[u];
            }
        } else {
            printf("    prefill: %.1f ms, %.2f M cycles\n", ms, g->cycles / 1e6);
        }
        while (c < count && pos[c] < p) c++;
        if (c < count && pos[c] == p) {
            const int32_t *want = rows + (size_t)c * vp;
            int d = 0;
            for (int i = 0; i < vp; i++) d += acc[i] != want[i];
            printf("    position %4d (%s%d): %d of %d logits differ\n", p, bucket ? "decode_" : "prefill ", bucket ? bucket : n, d, vp);
            diffs += (uint64_t)d;
            checked++;
        }
    }
    static const char *const unit[] = { "LOAD", "MATRIX", "VECTOR", "STORE" };
    for (int b = 0; b < m->n_buckets; b++) {
        if (!per[b].runs) continue;
        printf("    decode_%-4d %4d tokens: %6.1f ms, %6.2f M cycles, %6" PRIu64 " descriptors per token\n", m->bucket[b], per[b].runs,
               per[b].ms / per[b].runs, per[b].cycles / per[b].runs / 1e6, per[b].descs);
        printf("               ");
        for (unsigned u = 0; u < MINI_TPU_NUM_UNITS; u++)
            printf(" %s busy %4.1f %% wait %4.1f %%%s", unit[u], 100 * per[b].busy[u] / per[b].cycles,
                   100 * per[b].wait[u] / per[b].cycles, u + 1 < MINI_TPU_NUM_UNITS ? "," : "\n");
    }
    printf("%d positions checked, %" PRIu64 " logits differ\n%s\n", checked, diffs, diffs || !checked ? "FAILED" : "PASSED");
    free(ids); free(gold); free(acc);
    return diffs || !checked;
}

int main(int argc, char **argv)
{
    const char *fw_hex = NULL;
    int c, check_mode = 0, verbose = 0, last_pos = -1, timeout_s = 10;
    while ((c = getopt(argc, argv, "d:f:t:cm:v")) != -1) {
        switch (c) {
        case 'd': dir = optarg; break;
        case 'f': fw_hex = optarg; break;
        case 't': timeout_s = atoi(optarg); break;
        case 'c': check_mode = 1; break;
        case 'm': last_pos = atoi(optarg); break;
        case 'v': verbose = 1; break;
        default:
            fprintf(stderr, "usage: %s [-d dir] [-f fw.hex] [-t timeout_s] [-v] | -c [-m last_position]\n", argv[0]);
            return 2;
        }
    }
    static gen_t g;
    g.timeout_us = (uint64_t)timeout_s * 1000000u;
    if (load(&g, fw_hex)) return 1;
    int rc = 0;
    if (check_mode) rc = check(&g, last_pos);
    else serve(&g, verbose);
    fclose(g.embed);
    free(g.wpe); free(g.scale);
    mtpu_close(&g.dev);
    return rc;
}
