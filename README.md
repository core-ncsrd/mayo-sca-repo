# Side-Channel Analysis of the MAYO Secret Linear Map on a Cortex-M4

---


## Hardware

- **Target:** STM32F303 (ARM Cortex-M4) on a CW308 UFO board (`PLATFORM=CW308_STM32F3`)
- **Capture:** ChipWhisperer-Lite CW1173, ADC ×4, 7.37 MHz target clock
- **Toolchain:** `arm-none-eabi-gcc`, Python 3.10+, ChipWhisperer 6.x

The analysis scripts need **no hardware** — they run on the released traces.
Only the `tvla_*`/`cap_*`/`campaign.py` capture scripts need a physical target.

---

## Layout

```
firmware/     mayo_lm.c          unprotected O·x target
              mayo_lm_masked.c   masked only (two shares)   — still leaks, |t|=17.94
              mayo_lm_masked2.c  masked + row-shuffled      — the countermeasure
              simple_arithmetic.h, mem.h   GF(16) code, verbatim from MAYO-C
analysis/     cpa.py             CPA on row 0 (the attack)
              tvla_fvf.py        fixed-vs-fixed TVLA capture, unprotected
              tvla_masked.py     fixed-vs-fixed TVLA capture, protected
              cap_block.py       drift-free block-interleaved capture  ← the 3.62 result
              cap_interleaved.py per-trace interleaved variant
              campaign.py        cycle counts + overhead
              analyze_all.py     master post-hoc analysis → results.json
              analyze2.py        extended CPA (rows 0–5), naive-mask analysis
              fig_final.py       regenerates the paper figures
results/      *.json             every measured number in the paper
figures/      *.pdf, *.png       the four manuscript figures
docs/         EXPERIMENT_REPORT.md   full lab report
```

---

## Getting the traces

The raw traces are **852 MB** and are hosted on Zenodo, not in git:

> **Zenodo DOI:** `10.5281/zenodo.XXXXXXX` *(to be minted on publication)*

```bash
# after downloading and unpacking the Zenodo archive into ./
export MAYO_SCA_ROOT="$PWD"
```

Expected layout once unpacked:

| Directory | Contents | Shape |
|---|---|---|
| `traces/` | fixed-vs-random TVLA + CPA set | 1000 × 5000 |
| `traces_fvf/` | fixed-vs-fixed, unprotected | 2 × 500 × 20000 |
| `traces_masked/` | masked only | 2 × 1000 × 24000 |
| `traces_masked2/` | masked + shuffled, sequential capture | 2 × 800 × 24000 |
| `traces_masked2_bi/` | masked + shuffled, **block-interleaved** | 2 × 800 × 24000 |
| `traces_cpa/` | dedicated CPA capture | 2500 × 6000 |
| `traces_rand_O/` | random-`O` / fixed-`x` | 1969 × 20000 |

---

## Reproducing the results

```bash
pip install -r requirements.txt
export MAYO_SCA_ROOT="$PWD"     # defaults to the repo root if unset

# 1. The attack: CPA on row 0  → ranks [2,1,1,1,4,1,3,1], 24 candidates
python analysis/cpa.py

# 1b. The control: naive global-max CPA, showing the g=1 alias  → 1/8
python analysis/cpa.py --plain

# 2. Everything else: TVLA, chi-squared, t-vs-N, CPA  → results/results.json
python analysis/analyze_all.py

# 3. Extended CPA across rows + naive-mask analysis
python analysis/analyze2.py

# 4. Regenerate the paper figures
python analysis/fig_final.py
```

### With hardware attached

```bash
export CW_FIRMWAREPATH=/path/to/chipwhisperer/firmware/mcu

make -f firmware/Makefile             PLATFORM=CW308_STM32F3 CRYPTO_TARGET=NONE
make -f firmware/Makefile_masked2     PLATFORM=CW308_STM32F3 CRYPTO_TARGET=NONE

python analysis/tvla_fvf.py     # unprotected  → |t| = 195.7
python analysis/cap_block.py    # protected, drift-free → |t| = 3.62
python analysis/campaign.py     # cycle counts → 2.22×
```

---

## Two methodological notes

**Drift-free acquisition is required.** Capturing the two TVLA classes in
sequential blocks reports a misleading `|t| ≈ 9.72` for the protected firmware. A
same-secret control — splitting one class against itself, where no
secret-dependent leakage can exist — shows this is acquisition drift, not
leakage. `cap_block.py` interleaves the classes in blocks of 25 traces so both
traverse the same drift trajectory, and reports the correct `|t| = 3.62`. Any
countermeasure evaluated near the 4.5 gate needs this.

**The GF(16) identity rule matters.** Partial correlation annihilates the `g=1`
predictor, so a nibble whose true value *is* 1 becomes invisible to it. Such a
nibble is still identifiable without knowing the key — the plain spectrum peaks
at `g=1` while the partial spectrum is flat — and `cpa.py` and `analyze_all.py`
both apply that rule. Omitting it mis-ranks those nibbles and inflates the
reported key-space.

---

## Scope and ethics

This artifact targets a reference-faithful **research** implementation of the
MAYO secret linear map on the authors' own evaluation board. Full `mayo_sign`
does not fit the STM32F303's ~48 KB SRAM, so the linear map is isolated as a
reduced target using MAYO-C's own GF(16) code verbatim. It is intended for
implementation-security evaluation and countermeasure development, not for
attacking third-party systems.

## Licence

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
`firmware/simple_arithmetic.h` and `firmware/mem.h` are reproduced unmodified
from the MAYO reference implementation under the same licence.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Funding

Supported by the European Union's Horizon Europe programme under grant agreements
**No. 101225759 (PQ-NEXT)** and **No. 101119547 (PQ-REACT)**. Views and opinions
expressed are those of the authors only and do not necessarily reflect those of
the European Union.
