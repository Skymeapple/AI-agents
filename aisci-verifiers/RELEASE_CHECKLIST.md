# Release checklist (task 1.5) — waiting on Liam's approval

Status: package builds and installs cleanly; tests pass against the installed wheel. NOT published.

Before publishing:
- [ ] Liam approves releasing it publicly and picks the venue (GitHub repository and/or PyPI).
- [ ] Confirm the licence (Apache-2.0 assumed).
- [ ] Replace the placeholder controlled list in aisci/screen.py with a documented loading mechanism; ship no real list.
- [ ] Remove or clearly flag simulation modules as examples.
- [ ] Tag v0.1.0; attach results/ as release assets.

Exact next action once approved: create the repository, push this folder, then `python -m build && twine upload dist/*` if PyPI is chosen.
