# Sarah: The Non-Terminating agent

## Basic use:
    
You need to clone jelly-bean-world and dopamine into a libs folder. Then you use pipenv `pipenv install` in the root directory. Using python `3.9.10`.

Then to run a long running agent and measure the memory of jelly bean world. Use `./mem_stat.sh <steps>` where `<steps>` is the number of steps to run. If -1 is passed in the agent will run forever.


## Install dependencies

First run
```
git submodule update --init --recursive
cd libs/jelley-bean-world
git submodule update --init --recursive
cd ../..
```

to install all the submodules.

Next make sure you have installed python v3.9.10 (pyenv is recommended), and pipenv.

With pipenv installed run:
```
pipenv install
```

Which should create a new virtual env and install all the necessary packages. For atari games download the roms from:

```
http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html
```

extract and install with
```
ale-import-roms location/to/extracted/roms
```
