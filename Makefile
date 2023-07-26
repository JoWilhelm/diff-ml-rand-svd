# We do not actually use make. 
# Instead we use Hatch for managing the project.
# These commands are just convenient shortcuts.

.PHONY: build clean test

build:
	hatch build

clean:
	hatch clean

install:
	python -m pip install -e .

test:
	hatch run test

