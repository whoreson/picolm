#!/usr/bin/perl
# Generates .hunger_build.bat for MSYS2 environment.
# Usage: perl gen_hunger.pl VS_ROOT CUDA_ROOT MSVC_VER CUDA_ARCH src1.c src2.c ...
use strict;
use warnings;

my ($vs, $cu, $ver, $arch) = @ARGV[0..3];
my @srcs = @ARGV[4..$#ARGV];

# Convert forward slashes to backslashes for Windows
my $vbs = $vs; $vbs =~ s|/|\\|g;
my $cbs = $cu; $cbs =~ s|/|\\|g;

my $cl_exe = "\"$vbs\\VC\\Tools\\MSVC\\$ver\\bin\\Hostx64\\x64\\cl.exe\"";
my $link_exe = "\"$vbs\\VC\\Tools\\MSVC\\$ver\\bin\\Hostx64\\x64\\link.exe\"";
my $vcvars = "\"$vbs\\VC\\Auxiliary\\Build\\vcvars64.bat\"";
my $nvcc = "\"$cbs\\bin\\nvcc.exe\"";
my $cudart = "\"$cbs\\lib\\x64\\cudart.lib\"";

my $cf = "/O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I\"$cbs\\include\" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1";

my $cuflags = "-O3 --fmad=false -std=c++14 -DPICOLM_GPU=1 -DPICOLM_CUDA=1 " .
    "-DPICOLM_SSM_WARP_KERNEL_VALIDATED -DPICOLM_SSM_CHUNKED_GPU_VALIDATED " .
    "-ccbin $cl_exe " .
    "-Xcompiler \"/EHsc /O2 /W3 /std:c++14 /arch:AVX512 /wd4244 /wd4267 /wd4101 /wd4102 " .
    "/DPICOLM_AVX512=1 /DPICOLM_AVX2=1 /DPICOLM_AVX=1 /DPICOLM_SSE3=1 /DPICOLM_SSE2=1 /DPICOLM_FMA=1\" " .
    "-gencode arch=compute_89,code=$arch -c -I.";

my @cu_files = (
    ["backend_gpu_kernels_win.obj", "backend_gpu_kernels.cu"],
    ["backend_gpu_host_core_win.obj", "backend_gpu_host_core.cu"],
    ["backend_gpu_host_ssm_win.obj", "backend_gpu_host_ssm.cu"],
    ["backend_gpu_host_misc_win.obj", "backend_gpu_host_misc.cu"],
);

my @objs = map { my $x = $_; $x =~ s/\.c$/.obj/; $x } @srcs;
my @cu_objs = map { $_->[0] } @cu_files;

my @L;
push @L, '@echo off';
push @L, "del /q picolm.exe @objs @cu_objs 2^>nul";
push @L, "call $vcvars";
push @L, "set \"PATH=$cbs\\bin;%PATH%\"";

for my $pair (@cu_files) {
    push @L, "$nvcc $cuflags -o $pair->[0] $pair->[1]";
    push @L, 'if errorlevel 1 exit /b 1';
}

for my $src (@srcs) {
    push @L, "$cl_exe $cf $src";
    push @L, 'if errorlevel 1 exit /b 1';
}

push @L, "$link_exe /OUT:picolm.exe @objs @cu_objs $cudart ws2_32.lib advapi32.lib";
push @L, 'if errorlevel 1 exit /b 1';
push @L, "del /q @cu_objs 2^>nul";

open my $fh, '>:raw', '.hunger_build.bat' or die "Cannot open .hunger_build.bat: $!";
print $fh join("\r\n", @L) . "\r\n";
close $fh;

print "Generated .hunger_build.bat with " . scalar(@L) . " lines\n";
