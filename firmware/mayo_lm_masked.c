// SPDX-License-Identifier: Apache-2.0
/*
 * MAYO-1 secret linear-map — ADDITIVE-MASKED implementation
 * Platform: CW308_STM32F3 (STM32F303, Cortex-M4), SS_VER_1_1
 *
 * Masking scheme
 * --------------
 * O = O0 XOR O1 (GF(16) additive shares, re-randomised every 'p' call)
 * Exploit GF(16) F2-linearity: mul_f(a XOR b, x) = mul_f(a,x) XOR mul_f(b,x)
 * Therefore:  Ox = O*x = (O0 XOR O1)*x = O0*x XOR O1*x
 *
 * No ISW gadget needed — share separation at the MATRIX level is sufficient
 * because mul_f distributes over XOR in both arguments. Overhead ≈ 2× cycles.
 *
 * Security level: first-order. A second-order adversary could combine the two
 * mat_mul traces. For higher-order protection, byte-level ISW masking applies.
 *
 * Commands
 * --------
 *  'k' (9 bytes)  : load one row of secret O (unshared; same as unmasked target)
 *  'p' (8 bytes)  : compute Ox = O*x with fresh random mask under trigger
 *                   Mask is generated internally by a 32-bit LCG PRNG.
 *
 * Trigger scope
 * -------------
 * trigger_high() immediately before O0*x (share 0 mat_mul)
 * trigger_low()  immediately after  O1*x (share 1 mat_mul) and XOR reduction
 * Mask generation is OUTSIDE the trigger to isolate the masked computation.
 */

#include "hal.h"
#include <stdint.h>
#include <string.h>
#include "simpleserial.h"

volatile unsigned char unsigned_char_blocker = 0;
#include "simple_arithmetic.h"

#define PARAM_o  8
#define PARAM_v  78

/* Unmasked secret — loaded once via 'k' commands */
static uint8_t O [PARAM_v * PARAM_o];

/* Two shares (rebuilt every computation) */
static uint8_t O0[PARAM_v * PARAM_o];
static uint8_t O1[PARAM_v * PARAM_o];

/* Output */
static uint8_t Ox0[PARAM_v];
static uint8_t Ox1[PARAM_v];
static uint8_t Ox [PARAM_v];

/* ---- 32-bit LCG PRNG (Numerical Recipes constants) ---- */
static uint32_t prng_state = 0xCAFEBABEu;

static uint8_t prng_nibble(void) {
    prng_state = prng_state * 1664525u + 1013904223u;
    return (uint8_t)((prng_state >> 24) & 0x0Fu);
}

/* ---- 'k' command — load one row of O (identical to unmasked target) ---- */
uint8_t load_key_row(uint8_t *buf, uint8_t len)
{
    if (len != (uint8_t)(PARAM_o + 1)) return 0x01;
    uint8_t row = buf[0];
    if (row >= (uint8_t)PARAM_v)       return 0x02;

    uint8_t *dst = O + (unsigned)row * PARAM_o;
    for (uint8_t j = 0; j < PARAM_o; j++)
        dst[j] = buf[1u + j] & 0x0Fu;
    return 0x00;
}

/* ---- 'p' command — masked computation ---- */
uint8_t compute_lm_masked(uint8_t *x, uint8_t len)
{
    (void)len;
    for (uint8_t j = 0; j < PARAM_o; j++)
        x[j] &= 0x0Fu;

    /*
     * Generate fresh random mask OUTSIDE the trigger.
     * O0[i] = random nibble,  O1[i] = O[i] XOR O0[i].
     */
    for (unsigned i = 0; i < (unsigned)(PARAM_v * PARAM_o); i++) {
        O0[i] = prng_nibble();
        O1[i] = O[i] ^ O0[i];
    }

    /* --- Masked computation under trigger --- */
    trigger_high();
    __asm__ volatile("" ::: "memory");

    /* Share 0 */
    mat_mul(O0, x, Ox0, PARAM_o, PARAM_v, 1);

    /* Share 1 */
    mat_mul(O1, x, Ox1, PARAM_o, PARAM_v, 1);

    /* Recombine */
    for (uint8_t i = 0; i < (uint8_t)PARAM_v; i++)
        Ox[i] = Ox0[i] ^ Ox1[i];

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

    /* Initialise O with deterministic test key so device is ready before 'k' */
    for (unsigned i = 0; i < (unsigned)(PARAM_v * PARAM_o); i++)
        O[i] = (uint8_t)((7u * i + 3u) & 0x0Fu);

    simpleserial_init();
    simpleserial_addcmd('k', PARAM_o + 1, load_key_row);
    simpleserial_addcmd('p', PARAM_o,     compute_lm_masked);

    while (1)
        simpleserial_get();
}
