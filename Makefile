# We do not actually use make. 
# Instead we use Hatch for managing the project.
# These commands are just convenient shortcuts.

.PHONY: build clean test install bachelier data fmt

build:
	hatch build

clean:
	hatch clean

install:
	python -m pip install -e .

test:
	hatch -e test run test

bachelier:
	hatch -e example run python examples/bachelier/bachelier.py

data:
	hatch -e test run python tests/test_bachelier.py

fmt:
	hatch -e lint fmt

edit:
	hatch -e example run nvim
