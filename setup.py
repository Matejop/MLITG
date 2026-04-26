from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
   name="MLL",
   version="1.0.0",
   description="Simple ML model",
   licence="MIT",
   long_description=long_description,
   author="MF",
   author_email="matej.fibiger@gmail.com",
   packages=["MLL"],
   install_requires=["orjson", "typing"],
)