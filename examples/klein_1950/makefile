results.png : klein_model_i.py data.csv parameters.csv
	python klein_model_i.py

parameters.csv : estimate_equations.py data.csv
	python estimate_equations.py

data.csv : process_data.py TableF10-3.csv
	python process_data.py

TableF10-3.csv :
	curl -s -Lk -o TableF10-3.csv http://www.stern.nyu.edu/~wgreene/Text/Edition7/TableF10-3.csv

.PHONY : clean
clean :
	rm -f TableF10-3.csv data.csv parameters.csv results.png
