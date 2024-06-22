.PHONY: help
help :

.PHONY: test
test :
	pytest -x -v --cache-clear --cov=fsic --cov-branch && coverage report -m && coverage html

.PHONY: sandbox
sandbox :
	cd sandbox && make

.PHONY: examples
examples :
	cd examples && make

.PHONY: hashes
hashes :
	git show-ref --tags | cut -d ' ' -f 1 > hashes.txt
