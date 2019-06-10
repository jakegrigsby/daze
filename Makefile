user : 
	pip install .

dev : 
	pip install -e .

clean :
	-rm -rf experiments/data
