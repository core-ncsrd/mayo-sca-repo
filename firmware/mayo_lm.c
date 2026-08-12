// SPDX-License-Identifier: Apache-2.0
/*
 * MAYO-1 secret linear-map SCA target — v2
 * Platform: CW308_STM32F3 (STM32F303, Cortex-M4), SS_VER_1_1
 *
 * Commands
 * --------
 *  'k' (9 bytes)  : load one row of O
 *                   buf[0]   = row index  (0 .. PARAM_v-1 = 0..77)
 *                   buf[1..8]= row data   (8 GF(16) nibbles, one per byte)
 *                   Returns 'z' ACK on success, 'z' with non-zero on error.
 *                   Call 78 times to load the full 78×8 secret matrix O.
 *
 *  'p' (8 bytes)  : run mat_mul(O, x, Ox) under trigger
 *                   buf[0..7]= x_i (8 GF(16) nibbles, one per byte)
 *                   Returns 'r' + 16 bytes (first 16 rows of Ox).
 *
 * Fixed-vs-fixed TVLA protocol
 * ----------------------------
 * Host loads O_A with 78 'k' commands, captures N 'p' traces.
 * Host loads O_B with 78 'k' commands, captures N 'p' traces with the
 * IDENTICAL x sequence (same PRNG seed). Any |t|>4.5 on the secret O.
 */

#include "hal.h"
#include <stdint.h>
#include <string.h>
#include "simpleserial.h"

volatile unsigned char unsigned_char_blocker = 0;
#include "simple_arithmetic.h"   /* mul_f, add_f, lincomb, mat_mul (MAYO-C verbatim) */

/* ---- MAYO-1 parameters ---- */
#define PARAM_o  8     /* oil variables  (cols of O)    */
#define PARAM_v  78    /* vinegar vars   (rows of O)    */

/* Secret oil matrix O: PARAM_v rows × PARAM_o cols, GF(16) nibbles in [0..0xF].
 * Initialised to the known-deterministic test key at boot; overwritten by 'k'. */
static uint8_t O[PARAM_v * PARAM_o];

/* Output buffer for Ox = O * x */
static uint8_t Ox[PARAM_v];

/* Deterministic test key: O[i] = (7*i + 3) & 0xF.
 * Used as the default so the device is always in a known state on power-up,
 * even without a prior 'k' sequence. */
static void init_secret_O(void)
{
    for (unsigned i = 0; i < (unsigned)(PARAM_v * PARAM_o); i++)
        O[i] = (uint8_t)((7u * i + 3u) & 0x0Fu);
}

/*
 * 'k' command — load one row of O (9-byte payload)
 *   buf[0]    row index in [0, PARAM_v)
 *   buf[1..8] PARAM_o GF(16) nibbles for that row
 * Returns 0x00 on success, 0x01 on bad length, 0x02 on out-of-range index.
 */
uint8_t load_key_row(uint8_t *buf, uint8_t len)
{
    if (len != (uint8_t)(PARAM_o + 1)) return 0x01;

    uint8_t row = buf[0];
    if (row >= (uint8_t)PARAM_v) return 0x02;

    uint8_t *dst = O + (unsigned)row * PARAM_o;
    for (uint8_t j = 0; j < PARAM_o; j++)
        dst[j] = buf[1u + j] & 0x0Fu;

    return 0x00;
}

/*
 * 'p' command — compute Ox = O * x under trigger (8-byte payload)
 *   buf[0..7]  x_i, PARAM_o GF(16) nibbles
 * Returns 'r' + 16 bytes (Ox[0..15]).
 *
 * Trigger brackets mat_mul exactly — the same call that appears at
 * mayo.c:486 in the reference implementation.
 */
uint8_t compute_lm(uint8_t *x, uint8_t len)
{
    (void)len;

    for (uint8_t j = 0; j < PARAM_o; j++)
        x[j] &= 0x0Fu;

    trigger_high();
    __asm__ volatile("" ::: "memory");
    mat_mul(O, x, Ox, PARAM_o, PARAM_v, 1);
    __asm__ volatile("" ::: "memory");
    trigger_low();

    simpleserial_put('r', 16, Ox);
    return 0x00;
}

int main(void)
{
    platform_init();
    init_uart();
    trigger_setup();

    init_secret_O();

    simpleserial_init();
    simpleserial_addcmd('k', PARAM_o + 1, load_key_row);  /* 9 bytes: row_idx + row_data */
    simpleserial_addcmd('p', PARAM_o,     compute_lm);    /* 8 bytes: x_i                */

    while (1)
        simpleserial_get();
}
