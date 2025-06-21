from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="YOLO11 example",          # Package name (e.g., "myproject")
    version="0.0.1",                   # Version (semantic versioning)
    packages=find_packages(),          # Automatically discover packages
    install_requires=requirements,     # Dependencies from requirements.txt
    python_requires=">=3.11",           # Minimum Python version
    author="Ilya Bychkov",                # Author name
    author_email='',
    description="Small educational program that gives you simple gui interface for YOLO11 object detection model",   # Brief summary
    url="https://github.com/ilya205/yolo11-detection-simple-project-example", # Project URL (optional)
)
