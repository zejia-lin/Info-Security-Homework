
# CUDA Blind Watermark

## Quickstart

1. Clone this repository and enter.
2. Build the project (make sure OpenCV and CUDAToolkit can befound by cmake)
```sh
mkdir build && cd build
cmake ..
make
```
3. Play with the CLI
```sh
./WmCLI
> embed pic_path watermark_path output_path
> extract wm_rows wm_cols pic_path output_path
```

## Test

All test scripts can be found in `./demo`

- `test_gpu.py`: A wrapper for `WmCLI` to test all the images in `./demo/pic`
- `test_cpu.py`: A wrapper for `guofei-wm` to test all the images in `./demo/pic`
- `test_video.py`: Additional expiriement for video watermark embedding
- `test_tool.py`: Tools for attacking the image

The scripts can be simply run as:

```sh
mkdir ./demo/out \
      ./demo/out/attack \
      ./demo/out/embeded \
      ./demo/out/extracted \
      ./demo/out/thresh
python test_gpu.py
```
