callgraph.dot: auto_vs_finite.cpp
	clang++ -S -emit-llvm auto_vs_finite.cpp -o - | /usr/local/Cellar/llvm/3.6.2/bin/opt -analyze -dot-callgraph

callgraph.svg: callgraph.dot
	cat callgraph.dot | c++filt | sed 's,>,\\>,g; s,-\\>,->,g; s,<,\\<,g' | dot -Tsvg -ocallgraph.svg
