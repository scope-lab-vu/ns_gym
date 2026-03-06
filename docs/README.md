# Docs Refs

To build HTML docs 

```
sphinx-build -b html source _build
```

To serve locally

```
cd _build
python -m http.server
```

When you push to main on Github the docs will automatically be build for you. 
