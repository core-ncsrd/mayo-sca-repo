// SPDX-License-Identifier: Apache-2.0
/*
 * MAYO-1 secret linear-map — MASKED + SHUFFLED (first-order countermeasure)
 * Platform: CW308_STM32F3 (STM32F303, Cortex-M4), SS_VER_1_1
 *
 * Countermeasure = additive Boolean masking of the oil matrix + row shuffling.
 *   1. Masking:   O = O0 XOR O1,  O0 fresh-random per 'p' call.  GF(16) mul is
 *                 F2-linear in its first argument, so O*x = O0*x XOR O1*x.
 *   2. Shuffling: the v=78 output rows of each share pass are evaluated in an
 *                 INDEPENDENT random order (two Fisher-Yates permutations), so a
 *                 given time sample averages over a uniformly random row and any
 *                 residual per-row leak is diluted across 78 slots.
 * The two share passes are kept temporally separated (all of Ox0, then all of
 * Ox1) so O0[r] and O1[r] are never manipulated back-to-back (which would leak
 * HD(O0[r],O1[r]) = HW(O[r])). The shares are recombined OUTSIDE the trigger.
 *
 * Security: first order. Overhead is ~2x multiplies (two passes) plus indexing.
 * Mask + permutation generation is OUTSIDE the trigger.
 */

#include "hal.h"
#include <stdint.h>
#include <string.h>
#include "simpleserial.h"

volatile unsigned char unsigned_char_blocker = 0;
#include "simple_arithmetic.h"

#define PARAM_o  8
#define PARAM_v  78

static uint8_t O  [PARAM_v * PARAM_o];   /* unmasked secret, loaded via 'k' */
static uint8_t O0 [PARAM_v * PARAM_o];   /* share 0 (random) */
static uint8_t O1 [PARAM_v * PARAM_o];   /* share 1 = O ^ O0 */
static uint8_t Ox0[PARAM_v];
static uint8_t Ox1[PARAM_v];
static uint8_t Ox [PARAM_v];
static uint8_t perm0[PARAM_v];
static uint8_t perm1[PARAM_v];

/* ---- xorshift32 PRNG (better distribution than LCG for masks) ---- */
static uint32_t prng_state = 0xCAFEBABEu;
static inline uint32_t prng32(void) {
    uint32_t x = prng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    prng_state = x; return x;
}
static inline uint8_t prng_nibble(void) { return (uint8_t)(prng32() & 0x0Fu); }

static inline void fisher_yates(uint8_t *p) {
    for (uint8_t i = 0; i < PARAM_v; i++) p[i] = i;
    for (uint8_t i = PARAM_v - 1; i > 0; i--) {
        uint8_t j = (uint8_t)(prng32() % (uint32_t)(i + 1));
        uint8_t t = p[i]; p[i] = p[j]; p[j] = t;
    }
}

uint8_t load_key_row(uint8_t *buf, uint8_t len)
{
    if (len != (uint8_t)(PARAM_o + 1)) return 0x01;
    uint8_t row = buf[0];
    if (row >= (uint8_t)PARAM_v)       return 0x02;
    uint8_t *dst = O + (unsigned)row * PARAM_o;
    for (uint8_t j = 0; j < PARAM_o; j++) dst[j] = buf[1u + j] & 0x0Fu;
    return 0x00;
}

uint8_t compute_lm_ms(uint8_t *x, uint8_t len)
{
    (void)len;
    for (uint8_t j = 0; j < PARAM_o; j++) x[j] &= 0x0Fu;

    /* fresh shares + two independent shuffles (all OUTSIDE the trigger) */
    for (unsigned i = 0; i < (unsigned)(PARAM_v * PARAM_o); i++) {
        O0[i] = prng_nibble();
        O1[i] = O[i] ^ O0[i];
    }
    fisher_yates(perm0);
    fisher_yates(perm1);

    trigger_high();
    __asm__ volatile("" ::: "memory");
    /* pass 0: all share-0 rows, shuffled */
    for (uint8_t p = 0; p < PARAM_v; p++) {
        uint8_t r = perm0[p];
        Ox0[r] = lincomb(O0 + (unsigned)r * PARAM_o, x, PARAM_o, 1);
    }
    /* pass 1: all share-1 rows, independently shuffled */
    for (uint8_t p = 0; p < PARAM_v; p++) {
        uint8_t r = perm1[p];
        Ox1[r] = lincomb(O1 + (unsigned)r * PARAM_o, x, PARAM_o, 1);
    }
    __asm__ volatile("" ::: "memory");
    trigger_low();

    /* recombine OUTSIDE the measured window */
    for (uint8_t i = 0; i < PARAM_v; i++) Ox[i] = Ox0[i] ^ Ox1[i];

    simpleserial_put('r', 16, Ox);
    return 0x00;
}

int main(void)
{
    platform_init();
    init_uart();
    trigger_setup();

    for (unsigned i = 0; i < (unsigned)(PARAM_v * PARAM_o); i++)
        O[i] = (uint8_t)((7u * i + 3u) & 0x0Fu);

    simpleserial_init();
    simpleserial_addcmd('k', PARAM_o + 1, load_key_row);
    simpleserial_addcmd('p', PARAM_o,     compute_lm_ms);

    while (1)
        simpleserial_get();
}
