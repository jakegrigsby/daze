user : 
	pip install .

dev : 
	pip install -e .

test :
	-pytest tests/
	-rm -rf tests/saves*

clean :
	-rm -rf saves
