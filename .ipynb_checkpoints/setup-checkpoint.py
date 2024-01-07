import pybind11
from distutils.core import setup, Extension

ext_modules = [
    Extension(
        'SVD4Rec',
        ['SVD4Rec.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]

setup(
    name='SVD4Rec',
    version='0.0.1',
    author='rootinshik',
    author_email='user@user.ru',
    description='pybind11 extension',
    ext_modules=ext_modules,
    requires=['pybind11']
)