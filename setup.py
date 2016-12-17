from setuptools import setup

setup(
    name="windrose",
    version="0.0.1",
    author="Gabriele Calvo",
    author_email="gcalvo87@gmail.com",
    description="Simple windrose and wind distribution chart generator from raw data and frequency tables.",
    keywords="windrose wind rose analysis data weibull distribution",
    url="https://github.com/gabrielecalvo/windrose",
    packages=['windrose'],
    install_requires=['pandas']
)
