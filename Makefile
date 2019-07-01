user : 
	pip install .

dev : 
	pip install -e .

test :
	pytest tests/
	rm -rf saves

clean :
	-rm -rf saves
