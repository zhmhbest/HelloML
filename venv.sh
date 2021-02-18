#!/bin/bash

# 创建虚拟环境
if [ ! -d ./venv ]; then
    if [[ `whereis conda` =~ conda:\ .+ ]]; then
        echo y | conda create -p ./venv python=3.6.8
    elif [[ `whereis virtualenv` =~ virtualenv:\ .+ ]]; then
        virtualenv ./venv
    else
        echo Error; exit 1
    fi
fi

# 升级PIP
export PATH=./venv:./venv/Scripts:$PATH
python -m pip install --upgrade pip

# 下载依赖
if [ ! -e ./packages/ok ]; then
    echo Downloading...
    torch=https://download.pytorch.org/whl/torch_stable.html
    mirror=https://mirrors.aliyun.com/pypi/simple
    pip download -r requirements.txt -d packages -f $torch -i $mirror
    if [ 0 -eq $? ]; then touch ./packages/ok; fi
fi

# 安装依赖
if [ -e ./packages/ok ]; then
    echo Installing...
    pip install --no-warn-script-location -r requirements.txt --no-index -f packages
    pip list
fi
