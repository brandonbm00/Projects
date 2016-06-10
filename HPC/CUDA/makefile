mcquad: mcquad.o
	nvcc -g -G -o mcquad mcquad.o -lm
mcquad.o: mcquad.cu
	nvcc -g -G -c mcquad.cu -lm
heat: heat.o
	nvcc -g -G -o heat heat.o -lm
heat.o: heat.cu
	nvcc -g -G -c heat.cu -lm
heat2: heat2.o
	nvcc -g -G -o heat2  heat2.o -L/usr/local/magma/lib -lmagma -llapack -lblas
heat2.o: heat2.cu
	nvcc -g -G -c -DADD_ -I/usr/local/magma/include heat2.cu
clean:
	-rm mcquad
	-rm heat
	-rm heat2
	-rm *.o
cleanoutput:
	-rm *.o[0-9][0-9]*
	-rm *.e[0-9][0-9]*
	-rm *.out
