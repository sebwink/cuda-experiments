all: build
	@rm -f compile_commands.json 
	@cd build && cmake .. && make 
	@ln -s build/compile_commands.json compile_commands.json

build:
	@mkdir -p build

tree:
	@tree -I *build*

clean:
	@rm -rf *build*
	@rm -f compile_commands.json
	@rm -f cmake/CPM*.cmake


build@%: build
	@cd build && cmake .. && make $* 

run@%: 
	@build/src/$(shell echo $* | sed 's/\./\//g')
