user : 
	pip install .

dev : 
	pip install -e .

test :
	-pytest --cov=daze --cov-report term-missing tests/
	-rm -rf tests/saves*

clean :
	-rm -rf saves
