user : 
	pip install .

dev : 
	pip install -e .

test :
	-pytest --cov=deepzip tests/
	-rm -rf tests/saves*

clean :
	-rm -rf saves
