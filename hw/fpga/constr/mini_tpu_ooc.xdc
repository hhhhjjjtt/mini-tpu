# Out-of-context constraints for mini_tpu_top: 100 MHz (KV260 pl_clk0 is 99.999 MHz).
create_clock -name clk -period 10.000 [get_ports clk]

# I/O timing budget: AXI interconnect / PS registers on the far side.
set_input_delay  -clock clk 2.000 [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay -clock clk 2.000 [get_ports -filter {DIRECTION == OUT}]
