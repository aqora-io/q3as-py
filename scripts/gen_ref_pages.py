"""Generate the code reference pages and navigation."""

import importlib

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent

modules = mkdocs_gen_files.config["extra"]["modules"]

for module in modules:
    module_path = Path(*module.split(".")) / "__init__.py"
    m = importlib.import_module(module)

    for item in m.__all__:
        doc_path = Path("reference", module, item)
        file_path = doc_path.with_suffix(".md")
        nav[(module, item)] = Path(module, item).with_suffix(".md").as_posix()
        with mkdocs_gen_files.open(file_path, "w") as fd:
            fd.write(f"::: {module}.{item}")
        mkdocs_gen_files.set_edit_path(file_path, module_path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
