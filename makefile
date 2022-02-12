.PHONY: install
install :
	pip install -U .

.PHONY: requirements
requirements :
	pip install `find requirements/*.txt | xargs cat | paste -s -d ' '`

.PHONY: test
test :
	pytest -x -v --cache-clear --cov=fsic --cov-branch && coverage report -m && coverage html

.PHONY: examples
examples :
	cd examples/_cookbook && python fortran_engine.py && python progress_bar.py && python quickstart.py
	cd examples/almon_2017 && python ami.py
	cd examples/godley-lavoie_2007 && python 3_sim.py && python 4_pc.py && python 5_lp1.py && python 6_reg.py && python 6_open.py && python 7_bmw.py
	cd examples/klein_1950 && make
