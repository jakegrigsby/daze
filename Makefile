user : 
	pip install .

dev : 
	pip install -e .

test :
	pytest tests/

clean :
	-rm -rf tests/data
