# mini-tpu runtime, built and run on the KV260.
#   make load     load mini_tpu.bit through fpga_manager, insert u-dma-buf
#   make          build libmtpu, the bring-up test, run_prefill and generate
#   make run      bring-up test (ARGS="-t gemm -v" to pick tests, FW_HEX=fw.hex to load a file)
#   make prefill  run the compiled model and compare it with the simulator (ARGS="-n 3")
#   make decode   decode check: every bucket, bit for bit against the reference (ARGS="-m 200")
#   make chat     type a line, GPT-2 finishes it (ARGS="--help" for options)
#   make state    show the fpga_manager state
#   make dist     on the workstation: collect what the board needs into build/dist/mini-tpu/
#
# In the repo the shared headers come from ../../hw/interface and the firmware image from
# ../../hw/fw/build/fw_image.h; in a dist folder both sit next to this Makefile.

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra
LDLIBS  ?= -lm
BUILD   ?= build

INTERFACE    ?= $(firstword $(wildcard interface) ../../hw/interface)
FW_IMAGE_DIR ?= $(if $(wildcard fw_image.h),.,../../hw/fw/build)
CPPFLAGS     += -Ilibmtpu -I$(INTERFACE)

LIB_SRCS := libmtpu/mtpu.c libmtpu/mtpu_desc.c
LIB_OBJS := $(LIB_SRCS:libmtpu/%.c=$(BUILD)/%.o)
HDRS     := $(wildcard libmtpu/*.h $(INTERFACE)/*.h)
FW_IMAGE := $(wildcard $(FW_IMAGE_DIR)/fw_image.h)

BIT          ?= mini_tpu.bit
FW_NAME      := $(notdir $(BIT))
FW_HEX       ?=
ARGS         ?=
MODEL        ?= model

UDMABUF_KO   ?= /home/ubuntu/udmabuf/u-dma-buf.ko
UDMABUF_SIZE ?= 268435456
UDMABUF_SYS  := /sys/class/u-dma-buf/udmabuf0

.PHONY: all load bitstream udmabuf run prefill decode chat state dist clean

all: $(BUILD)/mtpu_test $(BUILD)/run_prefill $(BUILD)/generate

$(BUILD):
	mkdir -p $@

$(BUILD)/%.o: libmtpu/%.c $(HDRS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

$(BUILD)/libmtpu.a: $(LIB_OBJS)
	$(AR) rcs $@ $^

$(BUILD)/mtpu_test: tests/mtpu_test.c $(BUILD)/libmtpu.a $(HDRS) $(FW_IMAGE)
	$(CC) $(CPPFLAGS) -I$(FW_IMAGE_DIR) $(CFLAGS) -o $@ $< $(BUILD)/libmtpu.a $(LDLIBS)

$(BUILD)/run_prefill: run_prefill.c $(BUILD)/libmtpu.a $(HDRS) $(FW_IMAGE)
	$(CC) $(CPPFLAGS) -I$(FW_IMAGE_DIR) $(CFLAGS) -o $@ $< $(BUILD)/libmtpu.a $(LDLIBS)

$(BUILD)/generate: generate.c $(BUILD)/libmtpu.a $(HDRS) $(FW_IMAGE)
	$(CC) $(CPPFLAGS) -I$(FW_IMAGE_DIR) $(CFLAGS) -o $@ $< $(BUILD)/libmtpu.a $(LDLIBS)

load: bitstream udmabuf

bitstream:
	@test -f $(BIT) || { echo "ERROR: $(BIT) not found in $$(pwd)"; exit 1; }
	sudo cp -v $(BIT) /lib/firmware/$(FW_NAME)
	echo $(FW_NAME) | sudo tee /sys/class/fpga_manager/fpga0/firmware > /dev/null
	@state=$$(cat /sys/class/fpga_manager/fpga0/state); echo "fpga_manager: $$state"; \
	  test "$$state" = operating || { echo "ERROR: bitstream did not load"; exit 1; }

# the module only allocates at insert, so a smaller buffer from an earlier boot is replaced
udmabuf:
	@have=$$(cat $(UDMABUF_SYS)/size 2>/dev/null || echo 0); \
	if [ "$$have" -ge $(UDMABUF_SIZE) ]; then \
	  echo "udmabuf0 present, $$have bytes"; \
	else \
	  if [ "$$have" != 0 ]; then echo "udmabuf0 is $$have bytes, replacing it"; sudo rmmod u-dma-buf; fi; \
	  echo "inserting u-dma-buf ($(UDMABUF_SIZE) bytes)"; \
	  sudo insmod $(UDMABUF_KO) udmabuf0=$(UDMABUF_SIZE) || { grep -i cma /proc/meminfo; exit 1; }; \
	  test -e $(UDMABUF_SYS)/size || { echo "ERROR: no udmabuf0, see dmesg (is CMA large enough?)"; grep -i cma /proc/meminfo; exit 1; }; \
	fi

run: $(BUILD)/mtpu_test
	sudo $(BUILD)/mtpu_test $(if $(FW_HEX),-f $(FW_HEX)) $(ARGS)

prefill: $(BUILD)/run_prefill
	sudo $(BUILD)/run_prefill -d $(MODEL) $(if $(FW_HEX),-f $(FW_HEX)) $(ARGS)

decode: $(BUILD)/generate
	sudo $(BUILD)/generate -d $(MODEL) -c $(if $(FW_HEX),-f $(FW_HEX)) $(ARGS)

chat: $(BUILD)/generate
	python3 complete.py --model $(MODEL) $(ARGS)

state:
	@cat /sys/class/fpga_manager/fpga0/state

# ---- workstation: one folder to copy to the board (root `make dist` sets OUTPUT) ----
DIST    ?= $(BUILD)/dist/mini-tpu
HW      ?= ../../hw
BIT_OUT ?= $(HW)/fpga/build/kv260/out
OUTPUT  ?=

dist:
	@test -n "$(OUTPUT)" || { echo "ERROR: run make dist from the repository root"; exit 1; }
	$(MAKE) -C $(HW)/fw
	rm -rf $(DIST)
	mkdir -p $(DIST)/libmtpu $(DIST)/tests $(DIST)/interface $(DIST)/model
	cp Makefile README.md run_prefill.c generate.c complete.py gpt2_tokenizer.py $(DIST)/
	cp libmtpu/*.c libmtpu/*.h $(DIST)/libmtpu/
	cp tests/*.c $(DIST)/tests/
	cp $(HW)/interface/*.h $(DIST)/interface/
	cp $(HW)/fw/build/fw_image.h $(HW)/fw/build/fw.hex $(DIST)/
	cp $(OUTPUT)/*.bin $(OUTPUT)/model.txt $(OUTPUT)/tokenizer.json $(DIST)/model/
	@if [ -f $(BIT_OUT)/mini_tpu.bit ]; then cp $(BIT_OUT)/mini_tpu.bit $(DIST)/; \
	  else echo "note: no $(BIT_OUT)/mini_tpu.bit; copy the bitstream separately"; fi
	@if [ $(HW)/rtl/common/mini_tpu_config.svh -nt $(BIT_OUT)/mini_tpu.bit ]; then \
	  echo "WARNING: the bitstream is older than the hardware config: rebuild it (make bitstream)"; fi
	@echo "copy $(DIST)/ to the board (e.g. /home/ubuntu/mini-tpu/), then: make load && make && make run && make prefill && make decode"

clean:
	rm -rf $(BUILD)
