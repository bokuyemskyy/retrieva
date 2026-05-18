mkdir -p evals/git_docs

for cmd in add commit push pull restore checkout branch log status clone; do
    git help $cmd | col -b > evals/git_docs/$cmd.txt
done