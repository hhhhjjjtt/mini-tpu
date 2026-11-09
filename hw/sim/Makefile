# mini-tpu RTL simulation (iverilog). Run from hw/sim.
#   make sim_matrix                      # matrix unit against a reference GEMM
#   make sim_vector                      # vmem + matrix + vector end to end
#   make sim_system                      # mini_tpu_top + firmware + DDR model (builds ../fw)
#   make sim_blob BLOB=<dir>             # run a compiler blob, dump DDR (tb/tb_blob.sv)
#   make sim_matrix_timing               # per-tile cycle breakdown of the matrix unit
#   make sim_all                         # matrix, vector and system TBs

RTL_DIR := ../rtl
TOP_SRC := ../fpga/top/mini_tpu_top.v
FW_DIR  := ../fw

include $(RTL_DIR)/files.mk
SRCS := $(addprefix $(RTL_DIR)/,$(RTL_SRCS))

BUILD ?= build
IVERILOG ?= iverilog
VVP ?= vvp

# msys make may not forward the Windows TMP to native iverilog, which then
# falls back to a missing C:\TEMP. On msys (cygpath present) use the build dir.
ifneq ($(shell command -v cygpath 2>/dev/null),)
export TMP  := $(shell cygpath -w "$(abspath $(BUILD))")
export TEMP := $(TMP)
endif

.PHONY: sim_matrix sim_vector sim_system sim_blob sim_matrix_timing fw sim_all clean

sim_matrix:
	@mkdir -p $(BUILD)
	$(IVERILOG) -g2012 -I$(RTL_DIR)/common -s tb_matrix_unit -o $(BUILD)/tb_matrix_unit.out \
		tb/tb_matrix_unit.sv $(SRCS)
	$(VVP) -n $(BUILD)/tb_matrix_unit.out

sim_vector:
	@mkdir -p $(BUILD)
	$(IVERILOG) -g2012 -I$(RTL_DIR)/common -s tb_vector_path -o $(BUILD)/tb_vector_path.out \
		tb/tb_vector_path.sv $(SRCS)
	$(VVP) -n $(BUILD)/tb_vector_path.out

# where the matrix unit's cycles go per M-tile (timing only, ideal vmem / consumer)
sim_matrix_timing:
	@mkdir -p $(BUILD)
	$(IVERILOG) -g2012 -I$(RTL_DIR)/common -s tb_matrix_timing -o $(BUILD)/tb_matrix_timing.out \
		tb/tb_matrix_timing.sv $(SRCS)
	$(VVP) -n $(BUILD)/tb_matrix_timing.out

fw:
	$(MAKE) -C $(FW_DIR)

# tb_system loads ../fw/build/fw.hex (relative to this directory)
sim_system: fw
	@mkdir -p $(BUILD)
	$(IVERILOG) -g2012 -I$(RTL_DIR)/common -s tb_system -o $(BUILD)/tb_system.out \
		tb/tb_system.sv tb/axi_ddr_model.sv $(SRCS) $(TOP_SRC)
	$(VVP) -n $(BUILD)/tb_system.out

# run a program blob and dump DDR: make sim_blob BLOB=<dir> (see tb/tb_blob.sv)
sim_blob: fw
	@mkdir -p $(BUILD)
	$(IVERILOG) -g2012 -I$(RTL_DIR)/common -s tb_blob -o $(BUILD)/tb_blob.out 		tb/tb_blob.sv tb/axi_ddr_model.sv $(SRCS) $(TOP_SRC)
	$(VVP) -n $(BUILD)/tb_blob.out +blob=$(BLOB)

sim_all:
	$(MAKE) sim_matrix
	$(MAKE) sim_vector
	$(MAKE) sim_system

clean:
	rm -rf $(BUILD)
