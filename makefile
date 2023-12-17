.PHONY: help
help :

.PHONY: install
install :
	pip install -U .

.PHONY: requirements
requirements :
	pip install `find requirements/*.txt | xargs cat | paste -s -d ' '`

.PHONY: test
test :
	pytest -x -v --cache-clear --cov=fsic --cov-branch && coverage report -m && coverage html

.PHONY: sandbox
sandbox :
	cd sandbox && make

.PHONY: examples
examples :
	cd examples/_cookbook && make
	cd examples/almon_2017 && make
	cd examples/define && make
	cd examples/godley-lavoie_2007 && make
	cd examples/klein_1950 && make
	cd examples/macrosimulation && make

.PHONY: hashes
hashes :
	git show-ref --tags | cut -d ' ' -f 1 > hashes.txt
