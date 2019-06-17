user : 
	pip install .

dev : 
	pip install -e .

clean :
	-rm -rf tests/data
