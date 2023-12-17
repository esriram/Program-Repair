# How to download regressions4j
**Note:**  Regressions4j puts all of its downloaded data in the ```transfer_cache``` folder inside a ```miner_space folder```. For me, the path is /home/username/Documents/miner_space/transfer_cache. The scripts in this directory moves the downloaded data from the transfer cache folder to this directory. 


Start downloading by running the following. Replace "pathToMinerSpace" with the path to the miner_space. This is the ```/home/username/Documents``` part from the example above. Replace currentDirectoryPath with the current directory path, e.g. path/to/Bug-Dataset-Generation/src/regminer. In addition. you can turn on maven testing by uncommenting the code block in download.sh

```
./download.sh pathToMinerSpace currentDirectoryPath
```

<br> Find out more about regressions4j at [RegMiner](https://github.com/SongXueZhi/RegMiner)
