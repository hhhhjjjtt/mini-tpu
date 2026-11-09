# =============================================================================
# syn_ooc.tcl — out-of-context synthesis + implementation of mini_tpu_top
# =============================================================================
#
# Non-project flow on the KV260 part. Gives resource use and post-route
# timing for the accelerator alone (no PS, no block design, no I/O).
#
#   vivado -mode batch -source syn_ooc.tcl -tclargs <out_dir> <stage> <xdc> <src>...
#     stage: synth (stop after synthesis) or impl (place + route)
#
# Driven by fpga/syn/run_vivado.sh (sources from rtl/files.mk + fpga/top).
# =============================================================================

set out_dir [file normalize [lindex $argv 0]]
set stage   [lindex $argv 1]
set xdc     [lindex $argv 2]
set srcs    [lrange $argv 3 end]
set part    "xck26-sfvc784-2LV-c"

file mkdir $out_dir
set_part $part

foreach f $srcs {
    if {[string match "*.sv" $f]} { read_verilog -sv $f } else { read_verilog $f }
}
read_xdc -mode out_of_context $xdc

set inc [file dirname [lsearch -inline $srcs *bram.v]]
synth_design -top mini_tpu_top -part $part -mode out_of_context -flatten_hierarchy rebuilt -include_dirs $inc
write_checkpoint -force [file join $out_dir post_synth.dcp]
report_utilization -file [file join $out_dir util_synth.rpt]
report_utilization -hierarchical -hierarchical_depth 3 -file [file join $out_dir util_synth_hier.rpt]
report_timing_summary -max_paths 20 -file [file join $out_dir timing_synth.rpt]

if {$stage eq "impl"} {
    opt_design
    place_design
    phys_opt_design
    route_design
    write_checkpoint -force [file join $out_dir post_route.dcp]
    report_utilization -file [file join $out_dir util_impl.rpt]
    report_utilization -hierarchical -hierarchical_depth 3 -file [file join $out_dir util_impl_hier.rpt]
    report_timing_summary -max_paths 50 -file [file join $out_dir timing_impl.rpt]
    report_timing -max_paths 30 -sort_by group -path_type summary -file [file join $out_dir timing_impl_paths.rpt]
}
puts "=== syn_ooc done: $stage -> $out_dir"
