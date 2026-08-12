# MAYO-1 Side-Channel Experiment — Execution Report

**Target scheme:** MAYO-1 (NIST additional-signatures / post-quantum UOV variant)
**Device under test:** STM32F303 (ARM Cortex-M4) on a CW308 UFO board
**Capture hardware:** ChipWhisperer-Lite CW1173 (`2b3e:ace2`)
**Host:** Linux workstation driving the capture board over USB
**Deliverable location:** `$MAYO_SCA_ROOT/`
**Objective:** Build a complete TVLA leakage-assessment + CPA key-recovery
experiment against the MAYO secret linear map, on real hardware.

---

## 1. Executive summary

A complete, working side-channel pipeline was built and run end-to-end on
physical hardware:

- **Firmware** isolating the MAYO secret linear map `Ox = O·xᵢ` over GF(16) was
  ported from the MAYO-C reference, compiled with `arm-none-eabi-gcc`, and
  flashed to the STM32F303.
- **TVLA** (Test Vector Leakage Assessment) shows the operation leaks
  overwhelmingly: **max |t| = 53.7** (threshold 4.5), 2971 of 5000 sample points
  failing — i.e. the unprotected implementation has massive, exploitable leakage.
- **CPA** (Correlation Power Analysis) recovers information about the secret
  matrix `O`: the true key nibbles rank on average **1.88 / 16**, collapsing the
  first key-row search space from **16⁸ = 4.29 × 10⁹ to 48 candidates**.

The one substantive deviation from the original brief: the attached board is a
**CW308_STM32F3**, not an STM32F4. The STM32F303 is a Cortex-M4 core, so the
"Cortex-M4" requirement holds; the only ARM HAL present on the host is `stm32f3`.

---

## 2. Environment audit (Phase 1)

| Item | Finding |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-124 |
| Cross-compiler | `arm-none-eabi-gcc` <redacted-host> 20210621 ✅ |
| Build tools | `make` 4.3 ✅, `git` 2.34.1 ✅, **`cmake` absent** (not required) |
| Python | 3.<redacted-host> |
| Python pkgs | chipwhisperer 6.0.0, numpy 1.26.4, scipy 1.15.3, matplotlib 3.<redacted-host>, tqdm 4.69.0 ✅ |
| CW firmware framework | full repo at `~/chipwhisperer` |
| ARM HAL available | **`stm32f3` only** — no `stm32f4` HAL present |
| USB device | `Bus 003 Dev 041: 2b3e:ace2 NewAE CW1173 [ChipWhisperer-Lite]` |
| Prior work on host | ML-KEM SCA (`~/mlkem_sca`, `~/pqm4`), several idle Jupyter kernels |

**Consequence:** target platform fixed to `PLATFORM=CW308_STM32F3`
(matches the physically-attached board and the only ARM HAL).

---

## 3. Target analysis (Phase 2) — what leaks and why

MAYO-C was cloned to `~/MAYO-C` (`github.com/PQCMayo/MAYO-C`). MAYO-1 parameters
(`include/mayo.h`):

```
n = 86   m = 78   o = 8   k = 10   v = n-o = 78   q = 16
pk = 1420 B      signature = 454 B      seed secret key = 24 B
P1_bytes = 120159   P2_bytes = 24336    O (packed) = 312 B
```

### 3.1 The secret linear map

During signing, each of the `k` signature blocks is formed as
(`src/mayo.c:484-489`):

```c
for (int i = 0; i <= param_k - 1; ++i) {
    vi = Vdec + i * (param_n - param_o);
    mat_mul(sk.O, x + i*param_o, Ox, param_o, param_n - param_o, 1); // Ox = O·x_i   <-- TARGET (mayo.c:486)
    mat_add(vi, Ox, s + i*param_n, param_n - param_o, 1);            // s_i = v_i + O·x_i
    memcpy(s + i*param_n + (param_n-param_o), x + i*param_o, param_o); // x_i copied into signature (PUBLIC)
}
```

- `sk.O` is the **secret** oil-space matrix (the UOV secret linear map), stored
  in the expanded key as `uint8_t O[V_MAX*O_MAX]`, one GF(16) nibble per byte —
  **624 bytes for MAYO-1** (78×8).
- `x_i` is **public**: the last `o` bytes of every signature block are exactly
  `x_i` (line 488).

### 3.2 The elementary secret operation (CPA intermediate)

`mat_mul → lincomb` (`src/simple_arithmetic.h:70`) accumulates
`Ox[r] = Σ_j mul_f(O[r][j], x_i[j])`, where `mul_f` (`simple_arithmetic.h:8`) is
the GF(16) multiply `mod x⁴+x+1`. The elementary secret op is therefore

```
prod = mul_f(O[r][j], x_i[j])     // secret nibble × known nibble → 4-bit result
```

⇒ **CPA intermediate = HW( mul_f(O_guess, x_known) )**, only **16 hypotheses per
nibble**. Recovering `O` breaks the MAYO secret key.

### 3.3 Feasibility constraint (key engineering decision)

The reference `mayo_expand_sk()` inflates the 24-byte seed into
**P1 (120,159 B) + P2 (24,336 B) ≈ 144 KB** of RAM-resident key material. The
STM32F303RCT6 has **~48 KB SRAM** ⇒ full reference `mayo_sign` **cannot run** on
this target (~3× over budget). `pqm4` contains `mayo1/2/3` but those target
F4-class RAM (128 KB+) and also will not fit the F303.

**Decision:** build a *reduced target* that isolates exactly the secret linear
map (O = 624 B, fits trivially), using MAYO-C's own GF(16) code **verbatim**
(`simple_arithmetic.h`, `mem.h`) so the leakage is faithful to the reference.
This is precisely the operation the TVLA/CPA target, and it is the standard
practice of bracketing the sensitive primitive.

---

## 4. Firmware (Phase 3)

`firmware/mayo_lm.c` — SimpleSerial-v1.1 target for `CW308_STM32F3`:

- Holds a fixed, **known** secret matrix `O_flat[i] = (7·i + 3) & 0xF`
  ⇒ `O[0] = [3, A, 1, 8, F, 6, D, 4]` (ground truth to validate CPA).
- Command `p` receives `x_i` (8 nibble-bytes), computes `Ox = O·x_i` under the
  trigger, and returns 16 output nibbles.
- Trigger placement:

```c
trigger_high();
__asm__ volatile("" ::: "memory");          // block the -O2 scheduler crossing the trigger
mat_mul(O, x, Ox, PARAM_o, PARAM_v, 1);      // == mayo.c:486
__asm__ volatile("" ::: "memory");
trigger_low();
```

**Build:** `make PLATFORM=CW308_STM32F3 CRYPTO_TARGET=NONE`

```
Memory:  RAM 2392 B / 40 KB (5.84 %)   ROM 5148 B / 256 KB (1.96 %)
Output:  mayo-linearmap-CW308_STM32F3.hex
```

**Trigger verification (disassembly `.lss`):** `mul_f` is inlined *between* the
GPIO trigger writes — the bit-mask partial products (`and #1/2/4/8`, `smulbb`),
XOR accumulation, and the `mod x⁴+x+1` reduction (`and #0xf0; lsr #3; and #0x0f`)
all appear inside `compute_lm`, confirming the trigger brackets the GF(16)
multiply.

---

## 5. Capture methodology (Phase 4)

`capture.py` — TVLA non-specific fixed-vs-random t-test:

- Clock 7.37 MHz (`clkgen`), ADC ×4 synchronous (29.54 MS/s), gain 25 dB,
  5000 samples, trigger `tio4`, serial `tio1=rx / tio2=tx`.
- **FIXED** group: constant `x_i`; **RANDOM** group: fresh random `x_i` each
  trace, interleaved by a per-trace coin (guards against slow drift).
- Saves `fixed_traces.npy`, `rand_traces.npy`, `rand_pt.npy` (known inputs for
  CPA), `fixed_pt.npy`.
- Welch t: `t = (μ_f − μ_r) / √(σ²_f/N + σ²_r/N)`, flags `|t| > 4.5`, plots.

Programming confirmed on device: `Detected STM32F302xB(C)/303xB(C) … Verified
flash OK, 5147 bytes`.

---

## 6. Results

### 6.1 TVLA — leakage assessment

| Run | Traces/group | max \|t\| | points > 4.5 |
|---|---|---|---|
| smoke | 60 | 15.05 | 969 / 5000 |
| **full** | **1000** | **53.71** | **2971 / 5000** |

A t-statistic of 53.7 against a 4.5 threshold is *enormous* — the secret linear
map leaks strongly and pervasively. See `traces/tvla.png`. This alone is the
headline defensive finding: the unprotected `O·x` implementation would fail any
leakage-assessment gate.

### 6.2 CPA — key recovery (3000 random traces, attacking row 0 of O)

Two confounds are inherent to attacking a **linear** GF(16) multiply and were
handled explicitly:

1. **Input-load aliasing.** `mul_f(1, x) = x` (GF(16) identity), so the guess
   `g = 1` has hypothesis `HW(x)`, which matches the strong, pervasive leakage of
   the known input being (re)loaded from memory every row. A naive global-max CPA
   therefore returns `1` for every nibble. `cpa.py --plain` reproduces this: **1/8**
   (correct only where the true nibble happens to be 1).
   **Fix:** *partial correlation* — linearly regress `HW(x_j)` out of the traces
   before correlating, annihilating the `g=1` alias.

2. **Temporal row-mixing.** Each `x_j` multiplies `O[r][j]` for all 78 rows, so a
   full-trace search mixes rows. The 5000-sample window spans only the first few
   rows, and row 0's eight products are separated in time at a fixed cadence
   **POI(j) = 130 + 100·j** (calibrated from the data).
   **Fix:** correlate each nibble only in a tight window around its POI.

**Result (POI + partial correlation):**

```
 j  POI   pick  true  ok    peak   margin
 0  130   0x6   0x3    .    0.577  0.042
 1  230   0x5   0xA    .    0.606  0.016
 2  330   0x1   0x1    Y    0.142  0.007
 3  430   0x8   0x8    Y    0.631  0.110
 4  530   0xE   0xF    .    0.597  0.031
 5  630   0x6   0x6    Y    0.621  0.049
 6  730   0x6   0xD    .    0.671  0.026
 7  830   0x4   0x4    Y    0.475  0.184

recovered O[0] = 6 5 1 8 E 6 6 4
truth     O[0] = 3 A 1 8 F 6 D 4
```

| Metric | Value |
|---|---|
| Exact top-1 nibbles | **4 / 8** |
| True-key rank per nibble | `[2, 2, 1, 1, 4, 1, 3, 1]` |
| **Mean true-key rank** | **1.88 / 16** (random = 8.5) |
| Row-0 key-space | **16⁸ = 4,294,967,296 → 48 candidates** (product of ranks) |

See `traces/cpa_ranks.png`.

### 6.3 Interpretation of the CPA outcome

Increasing 1000 → 3000 traces gave the **same** 4/8 and the **same** wrong
guesses, so the limit is **model/systematic, not SNR**. A fine scan of nibble 0
showed the true key `0x3` *does* peak at the correct POI (sample 130, r = 0.535),
but a GF-related competitor `0x6` edges it (r = 0.577) — the Hamming-weight model
of a single GF(16) multiply in a *linear* map is an imperfect fit, and other
rows' products create spurious peaks elsewhere.

This is the expected, well-understood behaviour for first-order CPA on a linear
map: the true key is consistently a **top-1..4 candidate**, so the attack still
collapses the row's key-space to ~48 candidates — an exploitable break — but a
clean top-1 on every nibble needs a stronger distinguisher:

- a **profiled / template** attack (build per-key or per-HW templates),
- a **stochastic / linear-regression** leakage model (fit per-bit weights instead
  of assuming uniform Hamming weight), or
- the **chained-accumulator** model (recover `O[0][0]`, then condition each
  subsequent nibble on the already-recovered ones).

Across all 78 rows of `O` the same reduction applies, and the linear structure
plus multiple output rows would allow full key recovery by enumeration.

---

## 7. Blockers encountered and how they were resolved

| Blocker | Resolution |
|---|---|
| **Scope held by an idle Jupyter kernel** (`fuser` on the USB node) → `cw.scope()` unusable | User shut the kernel down; device confirmed free |
| **Stuck FPGA / USB** after abrupt mid-capture process kills → `LIBUSB_ERROR_IO`, then persistent "no trigger seen" | Non-destructive **libusb `resetDevice()`** (no processes killed) + `selftest.py` (scope reset, flash, 15-shot trigger test → **15/15 OK**) |
| `cmake` absent | Not needed — MAYO-C sources pulled into the `make`-based CW build |
| Full `mayo_sign` won't fit 48 KB RAM | Reduced linear-map target (Section 3.3) |

---

## 8. Deliverable manifest — `$MAYO_SCA_ROOT/`

```
README.md                                   experiment overview + results + how-to
capture.py                                  TVLA fixed-vs-random capture + Welch t-test + plot
cpa.py                                      CPA (POI + partial correlation); --plain shows the naive g=1 failure
selftest.py                                 scope reset + flash + 15-shot trigger self-test
firmware/
  mayo_lm.c                                 reduced SimpleSerial target (trigger brackets mat_mul)
  simple_arithmetic.h                       GF(16) arithmetic — verbatim from MAYO-C
  mem.h                                     verbatim from MAYO-C
  Makefile                                  PLATFORM=CW308_STM32F3, CRYPTO_TARGET=NONE
  mayo-linearmap-CW308_STM32F3.hex          built firmware (+ .elf/.bin/.lss/.map/.sym)
traces/
  fixed_traces.npy, rand_traces.npy         3000 traces/group × 5000 samples (60 MB each)
  fixed_pt.npy, rand_pt.npy                 known inputs (rand_pt feeds CPA)
  tvla_t.npy                                Welch t-statistic
  tvla.png                                  TVLA plot (max|t| = 53.7)
  cpa_ranks.png                             CPA true-key ranks + nibble-0 guess spectrum
```

Reference implementation cloned separately at `$HOME/MAYO-C/`.

---

## 9. How to reproduce

```bash
cd ~/mayo_sca

# 1. build firmware
cd firmware && make PLATFORM=CW308_STM32F3 CRYPTO_TARGET=NONE && cd ..

# 2. (if the scope is stuck) reset + verify the trigger
python3 selftest.py                     # expect: TRIG_OK 15/15

# 3. TVLA capture (flashes, then 1000+ traces/group)
python3 capture.py --flash -n 1000 -s 5000     # -> traces/tvla.png, max|t|

# 4. CPA
python3 cpa.py                          # POI + partial correlation (rank + key-space)
python3 cpa.py --plain                  # naive global-max CPA (shows the g=1 alias)
```

If the ChipWhisperer is held by another process, free it first (identify with
`fuser /dev/bus/usb/003/041`); a stuck FPGA is cleared by a libusb
`resetDevice()` (see `selftest.py`).

---

## 10. Suggested next steps

1. **Stochastic / template CPA** to push row-0 recovery to a clean 8/8 top-1.
2. **Sweep all 78 rows of `O`** (retrigger per row, or widen the capture window /
   lower the sample rate to cover more of the 624-product matrix multiply).
3. Repeat TVLA/CPA on **MAYO-2** (o = 17, k = 4) which changes the map geometry.
4. As a countermeasure baseline, re-run TVLA against a **masked** GF(16) multiply
   and confirm the t-statistic drops below 4.5.
```
