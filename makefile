src = GBDT/src
build_dir = build
target = boost
objects = $(build_dir)/main.o $(build_dir)/logger.o $(build_dir)/util.o $(build_dir)/gbdt.o $(build_dir)/regression_tree.o
cc = g++ -std=c++17 -O3 -fopenmp -m64

all: $(build_dir) $(target)

$(build_dir):
	mkdir -p $(build_dir)

$(target): $(objects)
	$(cc) -o $(target) $(objects)

$(build_dir)/main.o: $(src)/main.cpp $(src)/lib/util.h $(src)/lib/logger.h $(src)/lib/regression_tree.h $(src)/lib/gbdt.h
	$(cc) -c $(src)/main.cpp -o $(build_dir)/main.o
$(build_dir)/util.o: $(src)/lib/util.cpp $(src)/lib/util.h $(src)/lib/logger.h
	$(cc) -c $(src)/lib/util.cpp -o $(build_dir)/util.o
$(build_dir)/logger.o: $(src)/lib/logger.cpp $(src)/lib/logger.h
	$(cc) -c $(src)/lib/logger.cpp -o $(build_dir)/logger.o
$(build_dir)/regression_tree.o: $(src)/lib/regression_tree.cpp $(src)/lib/regression_tree.h $(src)/lib/util.h $(src)/lib/logger.h
	$(cc) -c $(src)/lib/regression_tree.cpp -o $(build_dir)/regression_tree.o
$(build_dir)/gbdt.o: $(src)/lib/gbdt.cpp $(src)/lib/regression_tree.h $(src)/lib/gbdt.h $(src)/lib/logger.h
	$(cc) -c $(src)/lib/gbdt.cpp -o $(build_dir)/gbdt.o

.PHONY: clean
clean:
	rm -rf $(build_dir) $(target)
