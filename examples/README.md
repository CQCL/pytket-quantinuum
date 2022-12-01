# pytket-quantinuum example notebooks

Here resides the example notebooks of the `pytket-quantinuum` extension.
The `*.ipynb` notebooks are generated from the scripts in `examples/python`
using the [p2j](https://github.com/remykarem/python2jupyter) tool.


## How to modify the notebooks

Any change should be done to the corresponding `.py` file in `examples/python`.
For example, in order to modify the `examples/Quantinuum_emulator.ipynb` notebook, you need
to change the `examples/python/Quantinuum_emulator.py`. After that, you can update the
actual notebook by running the `p2j` command (still using the same example):

```bash
p2j -o -t examples/Quantinuum_emulator.ipynb examples/python/Quantinuum_emulator.py
```

## Adding new notebooks

To add a new notebook:
1. Create a python file in the `examples/python` folder with the source code for the new notebook. 
2. Run the `p2j` command above to create the notebook
3. Add the notebook name to the `maintained-notebooks.txt` file.


## Embedding the quantinuum logo

Please note that if you generate the notebooks from the python files, they will not contain
the quantinuum logo observed at the top of the notebooks. The markdown to embed the logo
is located in the `logo_header_markdown.md` file. So, you need to copy the content
of that file and put it at the top of the python script if you want the notebook to
include the logo.

You can append it by running:
```bash
logo=`cat logo_header_markdown.md`
sed -i "1i${logo}" python/<generator_python_file>
```

:warning: Beware that when committing your changes to the repo, you must check-in the *updated notebook
including the logo header*, but check-in the *updated `<generator_python_file>` without the
markdown cell for the logo*. This is because the `check-examples` script used in the CI, already
appends the logo to the python scripts before generating a temporary notebook, and then this
notebook is compared against the one that already exists.
