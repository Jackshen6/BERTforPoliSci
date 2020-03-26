# %load makefile
MODULENAME = YOURPOJECTFOLDERHERE 

help:
	@echo ""
	

init:
	conda env create --prefix ./envs --file environment.yml

run:
	sbatch run_nn_validate.sb	

.PHONY: init run 

