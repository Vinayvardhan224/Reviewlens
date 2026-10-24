from setuptools import setup, find_packages

setup(
    name="reviewlens",
    version="1.0.0",
    description="ReviewLens: An NLP-Based Smart Product Review Analysis System",
    author="ReviewLens Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "nltk>=3.8.1",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.18.0",
        "wordcloud>=1.9.0",
        "streamlit>=1.30.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
        "beautifulsoup4>=4.12.0",
    ],
    entry_points={
        "console_scripts": [
            "reviewlens=cli:main",
        ],
    },
)
