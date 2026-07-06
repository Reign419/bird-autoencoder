SITE=$(python -c "import site; print(site.getsitepackages()[0])")

export LD_LIBRARY_PATH=$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/cuda_cupti/lib:$SITE/nvidia/cuda_nvrtc/lib:$SITE/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

export TF_CPP_MIN_LOG_LEVEL=1
