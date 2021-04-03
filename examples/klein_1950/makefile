results.png : klein_model_i.py data.csv parameters.csv
	python klein_model_i.py

parameters.csv : estimate_equations.py data.csv
	python estimate_equations.py

data.csv : process_data.py TableF10-3.csv
	python process_data.py

.PHONY : clean
clean :
	rm -f data.csv parameters.csv
