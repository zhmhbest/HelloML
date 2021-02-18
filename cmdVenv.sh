# pip install virtualenv
# pip install virtualenvwrapper
# pip install virtualenvwrapper-win
if [ ! -d venv ]; then
    echo 'make venv'
    if [ "Windows_NT" == "$OS" ]; then
        # Windows
        export WORKON_HOME=$(cygpath -w `pwd`)
        cmd /c mkvirtualenv venv
    else
        # Linux
        export WORKON_HOME=`pwd`
        mkvirtualenv venv
    fi
fi