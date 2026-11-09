# =============================================================================
# build_kv260.tcl — mini-tpu (mini_tpu_top) for the Kria KV260: project, block
# design, bitstream, XSA
# =============================================================================
#
# GUI:       Tools > Run Tcl Script... > select this file
# Terminal:  vivado -mode batch -nojournal -log fpga/build/kv260.log \
#                   -source fpga/build_kv260.tcl [-tclargs <jobs>]
#            (run from hw/; on Windows the executable is
#             D:\Xilinx\Vivado\2024.1\bin\vivado.bat)
#
#   jobs: parallel synthesis / implementation jobs (default: CPU count, max 8)
#   In the GUI, set it first in the Tcl console if needed: set mini_tpu_jobs 4
#
# Outputs (hw/fpga/build/kv260/):
#   project/mini_tpu_kv260.xpr   Vivado project (open it to inspect the block design)
#   out/mini_tpu.bit             bitstream (load on the board through fpga_manager)
#   out/mini_tpu.bit.bin         same, bootgen-processed (for xmutil / fpga_manager .bin flows)
#   out/mini_tpu.xsa             hardware platform (PetaLinux / Vitis / device tree)
#   out/fw.hex               controller firmware image, if ../fw has been built
#   out/address_map.txt      AXI address map
#   out/timing_summary.rpt, out/utilization.rpt, out/build_info.txt
#
# Block design:
#   PS (KV260 preset) M_AXI_HPM0_FPD -> SmartConnect -> mini_tpu s_axil (64 KB at 0xA000_0000)
#   mini_tpu m_axi_fetch / m_axi_load / m_axi_store -> SmartConnect -> PS S_AXI_HPC0_FPD (DDR)
#   pl_clk0 (~100 MHz) clocks everything; proc_sys_reset from pl_resetn0
#   mini_tpu irq -> pl_ps_irq0[0]
# HPC0 with AXI cache = 0000 is non-coherent, like the v1 DMA flow (udmabuf).
# The firmware is not part of the bitstream: the runtime loads fw.hex into
# I-mem through the AXI-Lite window while CORE_RESET = 1.
# =============================================================================

proc mini_tpu_log {msg} { puts "=== mini_tpu: $msg" }

# ---- paths ----
set fpga_dir [file normalize [file dirname [info script]]]
set hw_dir   [file normalize [file join $fpga_dir ..]]
set rtl_dir  [file join $hw_dir rtl]
set bld_dir  [file join $fpga_dir build kv260]
set proj_dir [file join $bld_dir project]
set out_dir  [file join $bld_dir out]
set proj_name mini_tpu_kv260
set bd_name   mini_tpu_bd

set part_name  "xck26-sfvc784-2LV-c"
set board_part "xilinx.com:kv260_som:part0:1.4"
set ctrl_base  0xA0000000

# ---- options (tclargs, or variables preset in the GUI console) ----
if {![info exists mini_tpu_jobs]} {
    if {[llength $argv] > 0} {
        set mini_tpu_jobs [lindex $argv 0]
    } elseif {[info exists ::env(NUMBER_OF_PROCESSORS)]} {
        set mini_tpu_jobs [expr {min(8, $::env(NUMBER_OF_PROCESSORS))}]
    } else {
        set mini_tpu_jobs 4
    }
}
mini_tpu_log "jobs=$mini_tpu_jobs"

# ---- sources from rtl/files.mk (same manifest as simulation) ----
proc mini_tpu_mk_lists {path} {
    set vars [dict create]
    set fh [open $path r]
    set cur ""
    while {[gets $fh line] >= 0} {
        set line [string trim [regsub {#.*$} $line ""]]
        if {$line eq ""} { set cur ""; continue }
        if {[regexp {^(\w+)\s*[:?]?=\s*(.*)$} $line -> name rest]} {
            set cur $name
            dict set vars $cur {}
            set line $rest
        }
        if {$cur eq ""} continue
        set cont [string match {*\\} $line]
        set line [string trimright $line "\\ "]
        foreach w $line { dict lappend vars $cur $w }
        if {!$cont} { set cur "" }
    }
    close $fh
    return $vars
}

set mk [mini_tpu_mk_lists [file join $rtl_dir files.mk]]
set srcs {}
foreach v [list COMMON_SRCS UNIT_SRCS] {
    if {![dict exists $mk $v]} { error "mini_tpu: $v not found in files.mk" }
    foreach f [dict get $mk $v] { lappend srcs [file join $rtl_dir $f] }
}
foreach f $srcs { if {![file exists $f]} { error "mini_tpu: missing source $f" } }
set top_src [file join $fpga_dir top mini_tpu_top.v]

# ---- project ----
if {![catch {current_project}]} { close_project }
file mkdir $out_dir
mini_tpu_log "create project $proj_dir/$proj_name.xpr"
create_project $proj_name $proj_dir -part $part_name -force
set_property board_part $board_part [current_project]
set_property target_language Verilog [current_project]

add_files -norecurse $srcs
add_files -norecurse $top_src
# module references need the include file in the project, not only an include dir. The
# module-reference parser also seems to look for it only next to plain Verilog (.v) sources:
# with only .sv files in rtl/common it fails to open it, so keep a .v file there (dual_port_ram.v).
set cfg_hdr [file join $rtl_dir common mini_tpu_config.svh]
add_files -norecurse $cfg_hdr
set_property file_type {Verilog Header} [get_files $cfg_hdr]
set_property include_dirs [list [file join $rtl_dir common]] [current_fileset]
foreach f $srcs {
    if {[string match *.sv $f]} { set_property file_type SystemVerilog [get_files $f] }
}
update_compile_order -fileset sources_1

# ---- block design ----
mini_tpu_log "block design $bd_name"
create_bd_design $bd_name

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} $ps
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
    CONFIG.PSU__USE__IRQ0      {1} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
] $ps

set npu  [create_bd_cell -type module -reference mini_tpu_top mini_tpu_0]
set rst  [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_pl0]
set smcc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc_ctrl]
set smcm [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc_mem]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $smcc
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] $smcm

# clocks and resets
set clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
connect_bd_net $clk \
    [get_bd_pins mini_tpu_0/clk] \
    [get_bd_pins rst_pl0/slowest_sync_clk] \
    [get_bd_pins smc_ctrl/aclk] \
    [get_bd_pins smc_mem/aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihpc0_fpd_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins rst_pl0/ext_reset_in]
connect_bd_net [get_bd_pins rst_pl0/peripheral_aresetn] [get_bd_pins mini_tpu_0/rst_n]
connect_bd_net [get_bd_pins rst_pl0/interconnect_aresetn] \
    [get_bd_pins smc_ctrl/aresetn] [get_bd_pins smc_mem/aresetn]

# control: HPM0_FPD -> SmartConnect (AXI4 128 b -> AXI-Lite 32 b) -> mini_tpu
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins smc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M00_AXI] [get_bd_intf_pins mini_tpu_0/s_axil]

# memory: fetch / LOAD / STORE masters -> SmartConnect -> HPC0 (DDR)
connect_bd_intf_net [get_bd_intf_pins mini_tpu_0/m_axi_fetch] [get_bd_intf_pins smc_mem/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins mini_tpu_0/m_axi_load]  [get_bd_intf_pins smc_mem/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins mini_tpu_0/m_axi_store] [get_bd_intf_pins smc_mem/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_mem/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]

# interrupt
connect_bd_net [get_bd_pins mini_tpu_0/irq] [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]

# addresses: control window at 0xA000_0000 (64 KB), DDR for the three masters
set ctrl_seg [get_bd_addr_segs -of_objects [get_bd_intf_pins mini_tpu_0/s_axil]]
assign_bd_address -offset $ctrl_base -range 64K \
    -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] $ctrl_seg
assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

# ---- wrapper ----
set bd_file [get_files $bd_name.bd]
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ---- synthesis, implementation, bitstream ----
mini_tpu_log "synthesis + implementation ($mini_tpu_jobs jobs); this takes a while"
launch_runs synth_1 -jobs $mini_tpu_jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    error "mini_tpu: synthesis failed, see $proj_dir/$proj_name.runs/synth_1/runme.log"
}
launch_runs impl_1 -to_step write_bitstream -jobs $mini_tpu_jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "mini_tpu: implementation failed, see $proj_dir/$proj_name.runs/impl_1/runme.log"
}

# ---- collect outputs ----
set impl_dir [get_property DIRECTORY [get_runs impl_1]]
set bit_src  [file join $impl_dir ${bd_name}_wrapper.bit]
file copy -force $bit_src [file join $out_dir mini_tpu.bit]

open_run impl_1
report_timing_summary -max_paths 20 -file [file join $out_dir timing_summary.rpt]
report_utilization -hierarchical -hierarchical_depth 3 -file [file join $out_dir utilization.rpt]
write_hw_platform -fixed -include_bit -force [file join $out_dir mini_tpu.xsa]
set wns [get_property STATS.WNS [get_runs impl_1]]
set whs [get_property STATS.WHS [get_runs impl_1]]

# address map
set fh [open [file join $out_dir address_map.txt] w]
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces -of_objects [get_bd_cells *]]] {
    set off [get_property OFFSET $seg]
    if {$off eq ""} continue
    puts $fh [format "%-60s %s  %s" $seg $off [get_property RANGE $seg]]
}
close $fh

# .bit.bin via bootgen (optional; the v1 board flow loads the .bit directly)
set bif [file join $out_dir mini_tpu.bif]
set fh [open $bif w]; puts $fh "all:\n{\n  [file join $out_dir mini_tpu.bit]\n}"; close $fh
set bootgen [file join $::env(XILINX_VIVADO) bin bootgen]
if {$::tcl_platform(platform) eq "windows"} { append bootgen ".bat" }
if {[catch {exec $bootgen -arch zynqmp -image $bif -process_bitstream bin -w on} msg]} {
    if {![file exists [file join $out_dir mini_tpu.bit.bin]]} {
        mini_tpu_log "WARNING: bootgen did not produce mini_tpu.bit.bin ($msg)"
    }
}

# firmware image
set fw_hex [file join $hw_dir fw build fw.hex]
if {[file exists $fw_hex]} {
    file copy -force $fw_hex [file join $out_dir fw.hex]
} else {
    mini_tpu_log "WARNING: $fw_hex not found; build it with 'make -C fw' and copy it next to the bitstream"
}

# build info
set fh [open [file join $out_dir build_info.txt] w]
puts $fh "built:       [clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]"
puts $fh "vivado:      [version -short]"
puts $fh "part:        $part_name ($board_part)"
puts $fh "ctrl window: $ctrl_base (64 KB, AXI-Lite on M_AXI_HPM0_FPD)"
puts $fh "masters:     fetch, load, store -> S_AXI_HPC0_FPD"
puts $fh "irq:         pl_ps_irq0\[0\]"
puts $fh "timing:      WNS $wns ns, WHS $whs ns"
if {![catch {exec git -C $hw_dir rev-parse --short HEAD} rev]} { puts $fh "git:         $rev (working tree may be uncommitted)" }
close $fh

mini_tpu_log "done: WNS $wns ns, WHS $whs ns"
if {$wns < 0 || $whs < 0} { mini_tpu_log "WARNING: timing NOT met, see out/timing_summary.rpt" }
mini_tpu_log "outputs in $out_dir"
