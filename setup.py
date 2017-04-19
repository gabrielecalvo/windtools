from setuptools import setup

setup(
    name="windtools",
    version="0.0.1",
    author="Gabriele Calvo",
    author_email="gcalvo87@gmail.com",
    description="Simple windtools and wind distribution chart generator from raw data and frequency tables.",
    keywords="windtools wind rose analysis data weibull distribution",
    url="https://github.com/gabrielecalvo/windtools",
    packages=['windtools'],
    install_requires=['pandas']
)
