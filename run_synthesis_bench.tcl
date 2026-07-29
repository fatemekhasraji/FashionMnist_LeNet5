# Synthesis Benchmark script for LeNet-5 Torus Accelerator across 3 FPGA devices

set devices {
    "Artix7"    "xc7a100tcsg324-1"
    "Zynq7000"  "xc7z020clg400-1"
    "Virtex7"   "xc7vx485tffg1157-1"
}

set src_files [list \
    "FashionMnist_LeNet5.srcs/sources_1/new/Controller.sv" \
    "FashionMnist_LeNet5.srcs/sources_1/new/PE.sv" \
    "FashionMnist_LeNet5.srcs/sources_1/new/SA.sv" \
    "FashionMnist_LeNet5.srcs/sources_1/new/accel_top.sv" \
]

file mkdir "synth_results"

foreach {dev_name part} $devices {
    puts "========================================================"
    puts " RUNNING SYNTHESIS FOR DEVICE: $dev_name ($part)"
    puts "========================================================"

    create_project -in_memory -part $part

    foreach f $src_files {
        read_verilog -sv $f
    }

    # Out-of-context synthesis for synthesizable accel_top datapath
    synth_design -top accel_top -part $part -mode out_of_context

    # Create 100MHz clock constraint (10ns period)
    create_clock -period 10.000 -name clk [get_ports clk]

    # Write reports
    report_utilization -file "synth_results/utilization_${dev_name}.txt"
    report_timing_summary -file "synth_results/timing_${dev_name}.txt"

    close_project
}

puts "SYNTHESIS BENCHMARK COMPLETED SUCCESSFULLY!"
