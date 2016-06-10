mcquad: mcquad.o func.o
	mpicc -o mcquad mcquad.o func.o -lm -std=gnu99
mcquad.o: mcquad.c func.h
	mpicc -c mcquad.c -lm -std=gnu99
func.o: func.c func.h
	mpicc -c func.c
heat2: heat2.o trid.o
	mpicc -o heat2 heat2.o trid.o -lm -std=c99 `pkg-config --libs lapack`
heat2.o: heat2.c trid.h
	mpicc -c heat2.c -lm -std=c99 `pkg-config --libs lapack`
trid.o: trid.c trid.h
	mpicc -c trid.c -std=c99
heat: heat.o
	mpicc -o heat heat.o -lm -std=c99
heat.o: heat.c
	mpicc -c heat.c -lm -std=c99	
clean:
	-rm mcquad
	-rm heat
	-rm heat2
	-rm *.o
cleanoutput:
	-rm *.o[0-9][0-9]*
	-rm *.e[0-9][0-9]*
	-rm *.out
