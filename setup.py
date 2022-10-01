from cx_Freeze import setup, Executable

setup(
    name = "Object Detection Software",
    version="0.1",
    description = "This software detects objects in real time",
    executables = [Executable("main.py")],
)