#!/bin/sh
# best-of-N generation speed for several q2k configurations, interleaved
# so background load hits them alike.
#   tools/bench.sh [-r runs] [-n tokens] -- "args A" "args B" ...
runs=3
ntok=64
prompt="def fibonacci(n):"
while [ $# -gt 0 ]; do
	case "$1" in
	-r) runs=$2; shift 2 ;;
	-n) ntok=$2; shift 2 ;;
	-p) prompt=$2; shift 2 ;;
	--) shift; break ;;
	*) break ;;
	esac
done
q2k=$(dirname "$0")/../q2k
tmp=$(mktemp -d)
i=0
while [ $i -lt "$runs" ]; do
	k=0
	for cfg in "$@"; do
		# shellcheck disable=SC2086
		$q2k $cfg -n "$ntok" --bench -p "$prompt" 2>&1 >/dev/null |
		    sed -n 's/.*gen: .*(\([0-9.]*\) tok\/s).*/\1/p' >> "$tmp/$k"
		k=$((k + 1))
	done
	i=$((i + 1))
done
k=0
for cfg in "$@"; do
	printf '%-40s best %6s  all %s\n' "[$cfg]" "$(sort -n "$tmp/$k" | tail -1)" \
	    "$(sort -n "$tmp/$k" | tr '\n' ' ')"
	k=$((k + 1))
done
rm -rf "$tmp"
