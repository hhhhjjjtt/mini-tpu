#!/usr/bin/env bash
# Out-of-context Vivado run of mini_tpu_top on the KV260 part.
#
#   fpga/syn/run_vivado.sh synth|impl      (path from hw/; runs from any directory)
#
# Vivado runs inside fpga/build/<stage>/, so reports, checkpoints and Vivado's
# own scratch files (.Xil, clockInfo.txt) all land there. Set VIVADO to override
# the executable.
#
# Not a make target on purpose: msys make on Windows passes child processes a
# stripped environment (no SystemRoot / TMP / XILINX_*), and vivado.bat
# cannot start without it.
set -euo pipefail

stage=${1:?usage: fpga/syn/run_vivado.sh synth|impl}
case $stage in synth|impl) ;; *) echo "stage must be synth or impl" >&2; exit 2;; esac

if [ -z "${VIVADO:-}" ]; then
    if command -v vivado >/dev/null 2>&1; then VIVADO=vivado
    elif [ -f /d/Xilinx/Vivado/2024.1/bin/vivado.bat ]; then VIVADO=/d/Xilinx/Vivado/2024.1/bin/vivado.bat
    else echo "vivado not found; set VIVADO" >&2; exit 2
    fi
fi

syn_dir=$(cd "$(dirname "$0")" && pwd)
hw=$(cd "$syn_dir/../.." && pwd)
out=$hw/fpga/build/${stage}
mkdir -p "$out"

# native Windows Vivado needs C:/... paths
win() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }

srcs=()
for f in $(make -s -C "$hw/rtl" print-srcs); do
    srcs+=("$(win "$hw/rtl/$f")")
done
srcs+=("$(win "$hw/fpga/top/mini_tpu_top.v")")

cd "$out"
"$VIVADO" -mode batch -nojournal -log vivado.log \
    -source "$(win "$syn_dir/syn_ooc.tcl")" \
    -tclargs "$(win "$out")" "$stage" "$(win "$hw/fpga/constr/mini_tpu_ooc.xdc")" "${srcs[@]}"
