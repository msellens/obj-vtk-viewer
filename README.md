# obj-vtk-viewer

## Installation

### Preinstallation (Mac)
* Verify that brew, uv, and conda have been installed

### Environment and Dependencies
* Create a new conda environment
```
conda create --name myenv python=3.12
conda activate myenv
```
* Install dependencies (I used uv for this)
```
uv pip install trame vtk trame-vuetify trame-vtk trame-simput
```
### Source code
```
git clone https://github.com/msellens/obj-vtk-viewer.git
cd obj-vtk-viewer
```
## Execution
```
python src/obj-vtk-viewer.py --port 1234
```
