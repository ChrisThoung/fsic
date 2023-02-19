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

.PHONY: examples
examples :
	cd examples/_cookbook && python aliases.py && python eval.py && python fortran_engine.py && python pandas_indexing.py && python progress_bar.py && python quickstart.py
	cd examples/almon_2017 && python ami.py
	cd examples/godley-lavoie_2007 && make
	cd examples/klein_1950 && make
	cd examples/define && python define_simple.py

.PHONY: hashes
hashes :
	git show-ref --tags | cut -d ' ' -f 1 > hashes.txt
