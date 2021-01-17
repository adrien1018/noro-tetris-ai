all:
	python tetris_internal_setup.py build_ext --inplace
cpp:
	g++ -Og -g -DDEBUG -I/usr/include/python3.9 -o tetris tetris.cpp
clean:
	rm -rf build *.so
