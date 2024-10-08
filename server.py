name: Python application

on:
  push:
    branches:
      - main  # Replace with your main branch name
  pull_request:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'  # Choose your Python version

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install torch torchvision transformers diffusers Flask

    - name: Run server
      run: |
        python server.py
