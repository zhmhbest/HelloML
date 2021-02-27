if [ -d venv ] && [ -d packages ]; then
    source "`pwd`/venv/Scripts/activate"
    pip install -r "requirements.txt" --no-index -f packages
fi