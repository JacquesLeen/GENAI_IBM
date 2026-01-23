install:
#install requirements
	pip install -r requirements.txt

markdown_lint:
	find notes -name '*.md' -exec markdownlint {} --fix \;