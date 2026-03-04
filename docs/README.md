# NS-Gym Website

The official website for NSGym, hosted on [nsgym.io](https://nsgym.io/).

## Running the Website
In your `nsgym` conda environment, do:
```
pip install -e ".[dev]"
```
Then, for Mac Users,
```
cd docs
make html
python -m http.server 8000 --directory build/html
```

For Windows,
```
cd docs
make.bat html
```
The website should be running on `http://localhost:8000`.