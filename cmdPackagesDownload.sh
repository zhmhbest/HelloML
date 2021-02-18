PYTORCH_LIST='https://download.pytorch.org/whl/torch_stable.html'
# PYPI_LIST='https://pypi.tuna.tsinghua.edu.cn/simple'
PYPI_LIST='https://mirrors.aliyun.com/pypi/simple'

pip download -r "requirements.txt" -d "packages" -f "$PYTORCH_LIST" -i "$PYPI_LIST"