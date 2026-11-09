# Source manifests for mini-tpu RTL (paths relative to hw/rtl/).

COMMON_SRCS := \
	common/dual_port_ram.v \
	common/unit_frontend.sv \
	common/queue_axi4_master_r.sv \
	common/sfifo.sv

# Units
UNIT_SRCS := \
	matrix_unit/systolic_array.sv \
	matrix_unit/act_transposer.sv \
	matrix_unit/matrix_unit.sv \
	vmem/vmem.sv \
	vector_unit/exp2neg.sv \
	vector_unit/lut_ram.sv \
	vector_unit/vector_unit.sv \
	dma/dma_burst_gen.sv \
	dma/load_unit.sv \
	dma/store_unit.sv \
	scalar_unit/picorv32.v \
	scalar_unit/issue_engine.sv \
	scalar_unit/scalar_unit.sv

RTL_SRCS := $(COMMON_SRCS) $(UNIT_SRCS)
