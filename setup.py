from setuptools import setup, find_packages


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename) as f:
        requirements = []
        for line in f:
            # Skip comments, -e, and -r lines
            if line.startswith('#') or line.startswith('-e') or line.startswith('-r'):
                continue
            # Remove whitespace and newlines
            requirement = line.strip()
            if requirement:  # Add non-empty lines
                requirements.append(requirement)
        return requirements


setup(
    name="yolo11-detection-simple-project-example",          # Package name (e.g., "myproject")
    version="0.0.1",                   # Version (semantic versioning)
    packages=find_packages(),          # Automatically discover packages
    install_requires=parse_requirements('requirements.txt'),     # Dependencies from requirements.txt
    python_requires=">=3.11",           # Minimum Python version
    author="Ilya Bychkov",                # Author name
    author_email='',
    description="Small educational program that gives you simple gui interface for YOLO11 object detection model",   # Brief summary
    url="https://github.com/ilya205/yolo11-detection-simple-project-example", # Project URL (optional)
)
