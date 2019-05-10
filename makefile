CXX = mpic++
CXXFLAGS = -march=native -ffast-math -std=c++11 -Wno-literal-suffix -fabi-version=0 -O3#-ggdb 
LFLAGS = -march=native -ffast-math -lm -std=c++11 -Wno-literal-suffix -fabi-version=0 -O3#-ggdb

Objects = main.o# snapshot.o aveRadical.o
RandomGen = mt19937-64.o
RandomObjs = #iterate.o initialization.o growth.o

lattice2d : $(Objects) $(RandomGen) $(RandomObjs)
	$(CXX) $(Objects) $(RandomGen) $(RandomObjs) -o simMIPS $(LFLAGS)

$(Objects) : config.h particles.h mt64.h
$(RandomGen) : mt64.h
$(RandomObjs) : #lattice2d.h mt64.hr

.PHONY : clean
clean :
	-rm *.o
