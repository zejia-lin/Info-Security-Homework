filepath=$1

filename=$(basename -- "$filepath")
extension="${filename##*.}"
filename="${filename%.*}"

_NVVM_BRANCH_=nvvm
_SPACE_= 
_CUDART_=cudart
_HERE_=/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin
_THERE_=/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin
_TARGET_SIZE_=
_TARGET_DIR_=
_TARGET_DIR_=targets/x86_64-linux
TOP=/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin/..
NVVMIR_LIBRARY_DIR=/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin/../nvvm/libdevice
LD_LIBRARY_PATH=/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin/../lib:/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/lib64:/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/libxml2-2.9.13-2zh52n5eucpu6yrqbyol4eivckwnpvve/lib:/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/zlib-1.2.12-fdetedch4vg4amja6odffwx33iqhxhgs/lib:/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/xz-5.2.5-ewneuymche4kdvfjh7x6ap34yjc7yahe/lib:/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/pkgconf-1.8.0-5snuei275rixsilkuf5ahanmvt7yn7wr/lib:/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/libiconv-1.16-ucjvuy6iohv6eyv2lgqxwbh3i5q2dnks/lib:/mnt/sda/2022-0526/public/wuk/v3/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.8.0-72ceydevxym52rxe7mefo7j6i3quvseb/targets/x86_64-linux/lib:
PATH=/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin/../nvvm/bin:/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin:/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin:/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/libxml2-2.9.13-2zh52n5eucpu6yrqbyol4eivckwnpvve/bin:/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/xz-5.2.5-ewneuymche4kdvfjh7x6ap34yjc7yahe/bin:/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/pkgconf-1.8.0-5snuei275rixsilkuf5ahanmvt7yn7wr/bin:/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/libiconv-1.16-ucjvuy6iohv6eyv2lgqxwbh3i5q2dnks/bin:/mnt/sda/2022-0526/home/lzj/downloads/spack/bin:/mnt/sda/2022-0526/public/wuk/v3/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.8.0-72ceydevxym52rxe7mefo7j6i3quvseb/bin:/mnt/sda/2022-0526/public/wuk/v3/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/libxml2-2.10.1-qus37pfc3hal4nuzo5p6x5t5jssl5hyu/bin:/mnt/sda/2022-0526/public/wuk/v3/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/xz-5.2.7-lq2apkoohncm7g253ptjpybqrti6pkax/bin:/mnt/sda/2022-0526/public/wuk/v3/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/pkgconf-1.8.0-ovzgo7o3kecaa4us2xxkkjnj6ly4mrzr/bin:/mnt/sda/2022-0526/public/wuk/v3/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/libiconv-1.16-btwruxkncnplq7g34irjudz2swjqcvba/bin:/mnt/sda/2022-0526/public/wuk/v3/spack/opt/spack/linux-debian11-x86_64/clang-11.0.1/gcc-7.5.0-trvywgosnstbqfa3jrgf5ilgslb6mzqk/bin:/mnt/sda/2022-0526/public/wuk/v3/spack/bin:/usr/local/bin:/usr/bin:/bin:/usr/games
INCLUDES="-I/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin/../targets/x86_64-linux/include"  
LIBRARIES="-L/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/targets/x86_64-linux/lib"
CUDAFE_FLAGS=
PTXAS_FLAGS=
gcc -D__CUDA_ARCH__=520 -D__CUDA_ARCH_LIST__=520 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  "-I/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=5 -D__CUDACC_VER_BUILD__=50 -D__CUDA_API_VER_MAJOR__=11 -D__CUDA_API_VER_MINOR__=5 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 "${filename}.cu" -o "/tmp/tmpxft_0018832b_00000000-9_${filename}.cpp1.ii" 
cicc --c++14 --gnu_version=70500 --display_error_number --orig_src_file_name "${filename}.cu" --orig_src_path_name "/mnt/sda/2022-0526/home/lzj/work/secure/aaa/src/${filename}.cu" --allow_managed   -arch compute_52 -m64 --no-version-ident -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "tmpxft_0018832b_00000000-3_${filename}.fatbin.c" -tused --gen_module_id_file --module_id_file_name "/tmp/tmpxft_0018832b_00000000-4_${filename}.module_id" --gen_c_file_name "/tmp/tmpxft_0018832b_00000000-6_${filename}.cudafe1.c" --stub_file_name "/tmp/tmpxft_0018832b_00000000-6_${filename}.cudafe1.stub.c" --gen_device_file_name "/tmp/tmpxft_0018832b_00000000-6_${filename}.cudafe1.gpu"  "/tmp/tmpxft_0018832b_00000000-9_${filename}.cpp1.ii" -o "/tmp/tmpxft_0018832b_00000000-6_${filename}.ptx"
ptxas -arch=sm_52 -m64  "/tmp/tmpxft_0018832b_00000000-6_${filename}.ptx"  -o "/tmp/tmpxft_0018832b_00000000-10_${filename}.sm_52.cubin" 
fatbinary -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=elf,sm=52,file=/tmp/tmpxft_0018832b_00000000-10_${filename}.sm_52.cubin" "--image3=kind=ptx,sm=52,file=/tmp/tmpxft_0018832b_00000000-6_${filename}.ptx" --embedded-fatbin="/tmp/tmpxft_0018832b_00000000-3_${filename}.fatbin.c" 
# rm /tmp/tmpxft_0018832b_00000000-3_${filename}.fatbin
gcc -D__CUDA_ARCH_LIST__=520 -E -x c++ -D__CUDACC__ -D__NVCC__  "-I/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=5 -D__CUDACC_VER_BUILD__=50 -D__CUDA_API_VER_MAJOR__=11 -D__CUDA_API_VER_MINOR__=5 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 "${filename}.cu" -o "/tmp/tmpxft_0018832b_00000000-5_${filename}.cpp4.ii" 
cudafe++ --c++14 --gnu_version=70500 --display_error_number --orig_src_file_name "${filename}.cu" --orig_src_path_name ${filepath} --allow_managed  --m64 --parse_templates --gen_c_file_name "/tmp/tmpxft_0018832b_00000000-6_${filename}.cudafe1.cpp" --stub_file_name "tmpxft_0018832b_00000000-6_${filename}.cudafe1.stub.c" --module_id_file_name "/tmp/tmpxft_0018832b_00000000-4_${filename}.module_id" "/tmp/tmpxft_0018832b_00000000-5_${filename}.cpp4.ii" 
gcc -D__CUDA_ARCH__=520 -D__CUDA_ARCH_LIST__=520 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS "-I/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin/../targets/x86_64-linux/include"   -m64 "/tmp/tmpxft_0018832b_00000000-6_${filename}.cudafe1.cpp" -o "/tmp/tmpxft_0018832b_00000000-11_${filename}.o" 
nvlink -m64 --arch=sm_52 --register-link-binaries="/tmp/tmpxft_0018832b_00000000-7_${filename}_dlink.reg.c"  -lcusolver -lcublas  "-L/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin/../targets/x86_64-linux/lib" "-L/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "/tmp/tmpxft_0018832b_00000000-11_${filename}.o"  -lcudadevrt  -o "/tmp/tmpxft_0018832b_00000000-12_${filename}_dlink.sm_52.cubin"
fatbinary -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " -link "--image3=kind=elf,sm=52,file=/tmp/tmpxft_0018832b_00000000-12_${filename}_dlink.sm_52.cubin" --embedded-fatbin="/tmp/tmpxft_0018832b_00000000-8_${filename}_dlink.fatbin.c" 
# rm /tmp/tmpxft_0018832b_00000000-8_${filename}_dlink.fatbin
gcc -D__CUDA_ARCH_LIST__=520 -c -x c++ -DFATBINFILE="\"/tmp/tmpxft_0018832b_00000000-8_${filename}_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"/tmp/tmpxft_0018832b_00000000-7_${filename}_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  "-I/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=5 -D__CUDACC_VER_BUILD__=50 -D__CUDA_API_VER_MAJOR__=11 -D__CUDA_API_VER_MINOR__=5 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -m64 "/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin/crt/link.stub" -o "/tmp/tmpxft_0018832b_00000000-13_${filename}_dlink.o" 
g++ -D__CUDA_ARCH_LIST__=520 -m64 -Wl,--start-group "/tmp/tmpxft_0018832b_00000000-13_${filename}_dlink.o" "/tmp/tmpxft_0018832b_00000000-11_${filename}.o" -lcusolver -lcublas  "-L/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin/../targets/x86_64-linux/lib" "-L/mnt/sda/2022-0526/home/lzj/downloads/spack/opt/spack/linux-debian11-zen/gcc-7.5.0/cuda-11.5.0-tc5i47ofl2udvmbpxonv2xikovyfpkkh/bin/../targets/x86_64-linux/lib"  -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group -o "../build/${filename}" 
