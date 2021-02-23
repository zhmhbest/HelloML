if [ -d venv ] && [ -d packages ]; then
    source "`pwd`/venv/Scripts/activate"
    # whereis pip
    pip install -r "requirements.txt" --no-index -f packages
fi