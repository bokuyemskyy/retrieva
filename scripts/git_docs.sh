mkdir -p data/git_docs

for cmd in add commit push pull restore checkout branch log status clone; do
    git help $cmd | col -b > data/raw/$cmd.txt
done