// mini-tpu — error codes (INTERFACE_SPEC.md §11), reported in STATUS[15:8].
#ifndef MINI_TPU_ERRORS_H
#define MINI_TPU_ERRORS_H

#define MINI_TPU_OK                 0x00u
#define MINI_TPU_ERR_BAD_MAGIC      0x01u
#define MINI_TPU_ERR_VERSION        0x02u
#define MINI_TPU_ERR_ARCH           0x03u
#define MINI_TPU_ERR_REGION_UNSET   0x04u
#define MINI_TPU_ERR_BAD_OPCODE     0x10u
#define MINI_TPU_ERR_BAD_FIELD      0x11u
#define MINI_TPU_ERR_LOOP_DEPTH     0x12u
#define MINI_TPU_ERR_LOOP_BODY      0x13u
#define MINI_TPU_ERR_CHUNK          0x14u
#define MINI_TPU_ERR_COUNTER_OVERFLOW   0x15u
#define MINI_TPU_ERR_FETCH_AXI      0x20u
#define MINI_TPU_ERR_UNIT_AXI       0x21u
#define MINI_TPU_ERR_VMEM_RANGE     0x22u
#define MINI_TPU_ERR_STREAM_MISMATCH    0x23u
#define MINI_TPU_ERR_WATCHDOG       0x30u
#define MINI_TPU_ERR_ABORTED        0x31u
#define MINI_TPU_ERR_CORE_TRAP      0x32u

#endif
