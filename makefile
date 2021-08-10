.PHONY: install
install :
	pip install -U .

.PHONY: test
test :
	pytest -x -v --cache-clear --cov=fsic --cov-branch && coverage report -m && coverage html
