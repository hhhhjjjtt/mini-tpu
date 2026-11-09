# mini-tpu build. Everything is driven by config.toml; see docs/BUILD.md.
#   make             compile the model into sw/compiler/output/
#   make check       run that output in the simulator against its golden data
#   make test        fast checks (about 15 s): simulator, tiny GPT, decode, tokenizer
#   make model       download the model and calibration text (once)
#   make sim         RTL regression
#   make bitstream   prints the Vivado command (msys make cannot start Vivado)
#   make dist        one folder to copy to the KV260, in sw/runtime/build/dist/
#   make clean       remove output/

# the repository's .venv (README.md), else python on PATH
PYTHON ?= $(firstword $(abspath $(wildcard .venv/Scripts/python.exe .venv/bin/python)) python)

export PYTHONPATH := sw/compiler
export HF_HOME := .cache

# Generates the headers now, on every make; they change only when [hardware] does.
PATHS := $(shell $(PYTHON) tools/gen_config.py --paths)
ifeq ($(words $(PATHS)),0)
$(error tools/gen_config.py failed)
endif
OUT   := $(word 1,$(PATHS))
CALIB := $(word 2,$(PATHS))
STAMP := $(word 3,$(PATHS))
PKG   := $(wildcard sw/compiler/mini_tpu/*.py)

.PHONY: all check test model sim bitstream dist clean
all: $(OUT)/manifest.json

# the stamp changes only with [model] and [build], so hardware edits do not recalibrate
$(CALIB): $(STAMP) sw/compiler/mini_tpu/calibrate.py
	$(PYTHON) -m mini_tpu.calibrate

$(OUT)/manifest.json: $(CALIB) config.toml $(PKG)
	$(PYTHON) -m mini_tpu.build

check: all
	$(PYTHON) -m mini_tpu.check

test:
	cd sw/compiler && $(PYTHON) -m unittest tests.test_sim tests.test_tiny_gpt tests.test_decode
	$(PYTHON) sw/runtime/tests/test_frontend.py

model:
	$(PYTHON) sw/compiler/model/fetch.py

sim:
	$(MAKE) -C hw/sim sim_all

# msys make gives child processes a stripped environment, and vivado.bat cannot start in it
bitstream:
	$(MAKE) -C hw/fw
	@echo "Run from hw/ in PowerShell:"
	@echo "  & D:\\Xilinx\\Vivado\\2024.1\\bin\\vivado.bat -mode batch -nojournal -log fpga/build/kv260.log -source fpga/build_kv260.tcl"

dist: all
	$(MAKE) -C sw/runtime dist OUTPUT=../../$(OUT)

clean:
	rm -rf sw/compiler/output
