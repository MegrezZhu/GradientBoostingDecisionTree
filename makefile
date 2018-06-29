src = GBDT/src
dist_dir = dist
obj_dir = $(dist_dir)/obj
objects = $(obj_dir)/main.o $(obj_dir)/logger.o $(obj_dir)/util.o $(obj_dir)/GBDT.o
target = $(dist_dir)/boost
cc = g++ -std=c++17 -O3 -fopenmp -lstdc++fs

$(target): $(objects)
	$(cc) -o $(target) $(objects)

$(obj_dir)/main.o: $(src)/main.cpp $(src)/lib/util.h $(src)/lib/logger.h $(src)/lib/GBDT.h
	$(cc) -c $(src)/main.cpp -o $(obj_dir)/main.o
$(obj_dir)/util.o: $(src)/lib/util.cpp $(src)/lib/util.h $(src)/lib/logger.h
	$(cc) -c $(src)/lib/util.cpp -o $(obj_dir)/util.o
$(obj_dir)/logger.o: $(src)/lib/logger.cpp $(src)/lib/logger.h
	$(cc) -c $(src)/lib/logger.cpp -o $(obj_dir)/logger.o
$(obj_dir)/GBDT.o: $(src)/lib/GBDT.cpp $(src)/lib/GBDT.h $(src)/lib/logger.h
	$(cc) -c $(src)/lib/GBDT.cpp -o $(obj_dir)/GBDT.o

.PHONY: clean
clean:
	rm $(objects) $(target) 2>/dev/null || true
