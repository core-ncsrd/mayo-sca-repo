#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NCSR "Demokritos" and AGH University of Krakow
"""Clean-reset the ChipWhisperer, flash the MAYO firmware, and self-test the
trigger before any long capture. Run on the remote host directly."""
import os, sys, time
import numpy as np
import chipwhisperer as cw

HERE = os.environ.get('MAYO_SCA_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ^ repository root (scripts live in analysis/); override with MAYO_SCA_ROOT.
FW = os.path.join(HERE, "firmware", "mayo-linearmap-CW308_STM32F3.hex")
FIXED_X = bytes([0x0A, 0x03, 0x0C, 0x05, 0x0F, 0x01, 0x07, 0x09])

def main():
    # Fresh connect; force an FPGA reload to clear any stuck armed state.
    scope = cw.scope()
    try:
        scope.reload_fpga() if hasattr(scope, "reload_fpga") else None
    except Exception as e:
        print("reload_fpga note:", e)
    scope.default_setup()
    scope.clock.clkgen_freq = 7.37e6
    try:
        scope.clock.adc_mul = 4
    except Exception:
        scope.clock.adc_src = "clkgen_x4"
    scope.adc.samples = 5000
    scope.gain.mode = "high"; scope.gain.db = 25
    scope.adc.basic_mode = "rising_edge"
    scope.trigger.triggers = "tio4"
    scope.io.tio1 = "serial_rx"; scope.io.tio2 = "serial_tx"

    target = cw.target(scope, cw.targets.SimpleSerial)

    print("[*] programming", os.path.basename(FW))
    cw.program_target(scope, cw.programmers.STM32FProgrammer, FW)

    # Explicit target reset so the MCU starts the freshly flashed firmware clean.
    scope.io.nrst = "low"; time.sleep(0.05)
    scope.io.nrst = "high_z"; time.sleep(0.2)
    target.flush()

    ok = 0
    N = 15
    peak = 0.0
    for i in range(N):
        scope.arm()
        target.simpleserial_write('p', bytearray(FIXED_X))
        timed_out = scope.capture()          # False == trigger seen
        r = target.simpleserial_read('r', 16, timeout=250)
        if not timed_out:
            ok += 1
            w = scope.get_last_trace()
            peak = max(peak, float(np.max(np.abs(w))))
    print(f"TRIG_OK {ok}/{N}  peak|sample|={peak:.3f}  resp_len={len(r) if r else 0}")
    scope.dis(); target.dis()
    sys.exit(0 if ok >= N - 2 else 3)

if __name__ == "__main__":
    main()
