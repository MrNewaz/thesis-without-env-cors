# Install Dependencies

```
pip install Flask==2.0.1 torchvision==0.10.0
```


# Download and put 'model.pth' file in the root directory

# Set Env 
```
FLASK_ENV=development FLASK_APP=main.py flask run
```

# Run Server

```
python -m flask run
```

# Post request to localhost:5000/predict and send image in the request body as a file