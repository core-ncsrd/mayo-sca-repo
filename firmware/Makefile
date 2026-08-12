# SPDX-License-Identifier: Apache-2.0
# Makefile for the MAYO-1 secret linear-map ChipWhisperer target.
#
# Build:
#     make PLATFORM=CW308_STM32F3 CRYPTO_TARGET=NONE
#
# Produces mayo-linearmap-CW308_STM32F3.hex  (flash with capture.py / cw.program_target).
#
# NOTE: STM32F303 is a Cortex-M4 core; CW308_STM32F3 is the only ARM HAL present
# on this host and matches the physically attached target board.

TARGET = mayo-linearmap
SRC += mayo_lm.c

# No ChipWhisperer crypto lib needed - GF(16) arithmetic comes from MAYO-C.
CRYPTO_TARGET = NONE

# SimpleSerial v1.1 (default). Set explicitly for clarity.
SS_VER = SS_VER_1_1

# Look in this directory for simple_arithmetic.h and mem.h (copied from MAYO-C).
EXTRAINCDIRS += .

# ChipWhisperer firmware framework on this host.
# Point this at your ChipWhisperer checkout, or export CW_FIRMWAREPATH.
FIRMWAREPATH ?= $(CW_FIRMWAREPATH)

# Makefile.inc pulls in hal/, simpleserial/ and crypto/ automatically.
include $(FIRMWAREPATH)/Makefile.inc
